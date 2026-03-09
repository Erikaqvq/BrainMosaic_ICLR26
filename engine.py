import math
import os
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import numpy as np
import json

import util.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        samples = utils.NestedTensor(samples, None) 
        targets = [
            {k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()}
            for t in targets
        ]

        outputs = model(samples)
        loss_dict = criterion(outputs, epoch, targets)
        losses = sum(loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        for k, v in loss_dict_reduced.items():
            metric_logger.update(**{k: v.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device,
             epoch=None,
             best_acc=None,
             best_ckpt_path=None,
             save_dir=None
             ):
    
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    total_correct = 0
    total_gt_labels = 0
    total_cosine = 0.0
    total_matched = 0
    total_unused_queries = 0
    samples_processed = 0

    all_topk_results = []
    global_idx = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = utils.NestedTensor(samples.to(device), None)
        targets = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        sentence_mode_logits = outputs["sentence_mode"]
        subjectivity_logits = outputs["subjectivity"]
        semantic_focus_logits = outputs["semantic_focus"]

        def _extract_labels(ts, key, dev):
            vals = []
            for t in ts:
                v = t[key]
                vals.append(int(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else int(v))
            return torch.tensor(vals, device=dev, dtype=torch.long)
        
        sentence_mode_tgt = _extract_labels(
            targets, "sentence_mode", sentence_mode_logits.device
        )
        subjectivity_tgt = _extract_labels(
            targets, "subjectivity", subjectivity_logits.device
        )
        semantic_focus_tgt = _extract_labels(
            targets, "semantic_focus", semantic_focus_logits.device
        )

        metric_logger.update(
            sentence_mode_acc=(
                (sentence_mode_logits.argmax(-1) == sentence_mode_tgt)
                .float()
                .mean()
                .item()
            ),
            subjectivity_acc=(
                (subjectivity_logits.argmax(-1) == subjectivity_tgt)
                .float()
                .mean()
                .item()
            ),
            semantic_focus_acc=(
                (semantic_focus_logits.argmax(-1) == semantic_focus_tgt)
                .float()
                .mean()
                .item()
            ),
        )

        processed = None
        if postprocessors and "topk" in postprocessors:
            processed = postprocessors["topk"].forward(outputs, epoch, targets)
            for i, pred in enumerate(processed):
                tgt_i = targets[i]
                def _get_scalar(t, key):
                    if key not in t: return None
                    v = t[key]
                    return int(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else int(v)
                
                all_topk_results.append({
                    "sample_index": global_idx,
                    "topk_words":  pred["topk_words"],
                    "topk_scores": pred["topk_scores"],
                    "sentence_mode_probs": pred["sentence_mode_probs"],
                    "subjectivity_probs": pred["subjectivity_probs"],
                    "semantic_focus_probs": pred["semantic_focus_probs"],
                    "gold_sentence": tgt_i.get("sentence", ""),
                    "gold_words":    tgt_i.get("words", []),
                    "gold_sentence_mode": _get_scalar(tgt_i, "sentence_mode"),
                    "gold_subjectivity": _get_scalar(tgt_i, "subjectivity"),
                    "gold_semantic_focus": _get_scalar(tgt_i, "semantic_focus"),
                })
                global_idx += 1

        loss_dict = criterion(outputs, epoch, targets)
        weight_dict = getattr(criterion, "weight_dict", {})
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_scaled   = {k: loss_dict_reduced[k] * weight_dict[k] for k in loss_dict_reduced if k in weight_dict}
        loss_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_scaled.values()) if loss_scaled else 0.0,
                             **loss_scaled, **loss_unscaled)
        
        indices = criterion.matcher(outputs, epoch, targets)
        B = outputs["pred_embeddings"].shape[0]
        Q = outputs["pred_embeddings"].shape[1]
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0 or len(tgt_idx) == 0:
                total_unused_queries += Q
                continue
            pred_embed = outputs["pred_embeddings"][i][src_idx]
            gt_embed   = targets[i]["word_embeddings"][tgt_idx]
            cosine_sim = F.cosine_similarity(pred_embed, gt_embed, dim=-1)

            total_cosine  += cosine_sim.sum().item()
            total_matched += cosine_sim.numel()
            total_correct += (cosine_sim > 0.82).sum().item()
            total_gt_labels += targets[i]["word_embeddings"].shape[0]
            total_unused_queries += (Q - len(src_idx))

        samples_processed += B

    metric_logger.synchronize_between_processes()
    matching_accuracy  = (total_correct / total_gt_labels) if total_gt_labels > 0 else 0.0
    mean_cosine        = (total_cosine  / total_matched)   if total_matched   > 0 else 0.0
    avg_unused_queries = (total_unused_queries / samples_processed) if samples_processed > 0 else 0.0

    sentence_mode_acc_g = metric_logger.meters.get("sentence_mode_acc", None)
    subjectivity_acc_g = metric_logger.meters.get("subjectivity_acc", None)
    semantic_focus_acc_g = metric_logger.meters.get("semantic_focus_acc", None)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({
        "matching_accuracy": matching_accuracy,
        "mean_cosine": mean_cosine,
        "avg_unused_queries": avg_unused_queries,
        "sentence_mode_acc": (
            sentence_mode_acc_g.global_avg if sentence_mode_acc_g else 0.0
        ),
        "subjectivity_acc": (
            subjectivity_acc_g.global_avg if subjectivity_acc_g else 0.0
        ),
        "semantic_focus_acc": (
            semantic_focus_acc_g.global_avg if semantic_focus_acc_g else 0.0
        ),
    })

    improved = False
    improved_epoch = None
    eps = 1e-8
    if utils.is_main_process() and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if (best_acc is None) or (matching_accuracy > best_acc + eps) or\
           (abs(matching_accuracy - (best_acc or 0.0)) <= eps and mean_cosine > 0.0):
            improved = True
            improved_epoch = epoch
            best_acc = matching_accuracy
            best_ckpt_path = os.path.join(save_dir, "best_by_matching_acc.pth")
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "model": model_to_save.state_dict(),
                "epoch": epoch,
                "best_matching_acc": float(best_acc),
                "mean_cos_at_best": float(mean_cosine),
            }, best_ckpt_path)
            
            with open(os.path.join(save_dir, "best_by_matching_acc.topk.json"), "w", encoding="utf-8") as f:
                json.dump(all_topk_results, f, ensure_ascii=False, indent=2)
            print(f"[Eval] New best matching_acc={best_acc:.4f} @ epoch={epoch} -> {best_ckpt_path}")

    return {
        "stats": stats,
        "topk": all_topk_results,
        "best_ckpt_path": best_ckpt_path,
        "best_updated": bool(improved), 
        "best_epoch": improved_epoch, 
    }

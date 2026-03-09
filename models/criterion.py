import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(z_pred, y_true, tau=0.07):
    z = F.normalize(z_pred, dim=-1)
    y = F.normalize(y_true, dim=-1)
    logits = (z @ y.t()) / tau
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, labels)

def compute_class_weights(counts):
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * len(counts)
    return weights

class SetCriterion(nn.Module):
    def __init__(self, *, matcher, 
                 num_classes=2, eos_coef=0.1, lambda_cls=0.5,
                 embed_loss="infonce", tau=0.07,
                 lambda_infonce=1.0, lambda_cos=1.0, 
                 lambda_sentence_mode=0.5, lambda_subjectivity=0.5, lambda_semantic_focus=0.5, lambda_sent=0.5,
                 sentence_mode_class_counts=None, subjectivity_class_counts=None, semantic_focus_class_counts=None):
        super().__init__()
        self.matcher = matcher

        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.lambda_cls = lambda_cls
        self.lambda_sentence_mode = lambda_sentence_mode
        self.lambda_subjectivity = lambda_subjectivity
        self.lambda_semantic_focus = lambda_semantic_focus
        self.embed_loss = embed_loss
        self.tau = tau
        self.lambda_infonce = lambda_infonce
        self.lambda_cos = lambda_cos
        self.lambda_sent = lambda_sent

        sentence_mode_counts = (
            torch.tensor(sentence_mode_class_counts, dtype=torch.float32)
            if sentence_mode_class_counts is not None
            else torch.ones(4, dtype=torch.float32)
        )
        subjectivity_counts = (
            torch.tensor(subjectivity_class_counts, dtype=torch.float32)
            if subjectivity_class_counts is not None
            else torch.ones(2, dtype=torch.float32)
        )
        semantic_focus_counts = (
            torch.tensor(semantic_focus_class_counts, dtype=torch.float32)
            if semantic_focus_class_counts is not None
            else torch.ones(5, dtype=torch.float32)
        )
        if sentence_mode_counts.numel() != 4:
            raise ValueError("sentence_mode_class_counts must have 4 values")
        if subjectivity_counts.numel() != 2:
            raise ValueError("subjectivity_class_counts must have 2 values")
        if semantic_focus_counts.numel() != 5:
            raise ValueError("semantic_focus_class_counts must have 5 values")
        
        self.semantic_focus_weights = compute_class_weights(semantic_focus_counts)
        self.subjectivity_weights = compute_class_weights(subjectivity_counts)
        self.sentence_mode_weights = compute_class_weights(sentence_mode_counts)
        self.register_buffer("semantic_focus_weights_buf", self.semantic_focus_weights)
        self.register_buffer("subjectivity_weights_buf", self.subjectivity_weights)
        self.register_buffer("sentence_mode_weights_buf", self.sentence_mode_weights)

        empty_weight = torch.ones(num_classes)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_sentence_embedding(self, outputs, targets):
        if self.lambda_sent <= 0:
            zero = outputs["pred_embeddings"].sum() * 0.0
            return {"loss_sent": zero}

        pred = outputs["pred_embeddings"][:, 0, :]
        tgt = torch.stack([t["sentence_embedding"] for t in targets]) 

        z = F.normalize(pred, dim=-1)
        y = F.normalize(tgt, dim=-1)

        loss_sent = 1.0 - (z * y).sum(dim=-1).mean()
        return {"loss_sent": loss_sent * self.lambda_sent}

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"] 
        B, N, _ = src_logits.shape
        target_classes = src_logits.new_full((B, N), 0, dtype=torch.long)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            target_classes[b, src_idx] = 1
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  weight=self.empty_weight)
        return {"loss_ce": loss_ce * self.lambda_cls}

    def _extract_labels(self, targets, key, device):
        vals = []
        for t in targets:
            v = t[key]
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(f"Target '{key}' must be scalar, got shape {tuple(v.shape)}")
                vals.append(int(v.detach().cpu().item()))
            else:
                vals.append(int(v))
        return torch.tensor(vals, device=device, dtype=torch.long) 

    def loss_attributes(self, outputs, targets):
        sentence_mode_logits = outputs["sentence_mode"]
        subjectivity_logits = outputs["subjectivity"]
        semantic_focus_logits = outputs["semantic_focus"]

        device = sentence_mode_logits.device

        sentence_mode_tgt = self._extract_labels(targets, "sentence_mode", device)
        subjectivity_tgt = self._extract_labels(targets, "subjectivity", device)
        semantic_focus_tgt = self._extract_labels(targets, "semantic_focus", device)
        
        loss_sentence_mode = F.cross_entropy(
            sentence_mode_logits,
            sentence_mode_tgt,
            weight=self.sentence_mode_weights_buf.to(device),
        ) * self.lambda_sentence_mode

        loss_subjectivity = F.cross_entropy(
            subjectivity_logits,
            subjectivity_tgt,
            weight=self.subjectivity_weights_buf.to(device),
        ) * self.lambda_subjectivity

        loss_semantic_focus = F.cross_entropy(
            semantic_focus_logits,
            semantic_focus_tgt,
            weight=self.semantic_focus_weights_buf.to(device),
        ) * self.lambda_semantic_focus

        return {
            "loss_sentence_mode": loss_sentence_mode,
            "loss_subjectivity": loss_subjectivity,
            "loss_semantic_focus": loss_semantic_focus,
        }

    def loss_embeddings(self, outputs, targets, indices):
        pred = outputs["pred_embeddings"] 
        matched_pred, matched_tgt = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            z = F.normalize(pred[b, src_idx], dim=-1)
            y = F.normalize(targets[b]["word_embeddings"][tgt_idx], dim=-1)
            matched_pred.append(z)
            matched_tgt.append(y)

        if len(matched_pred) == 0:
            zero = pred.sum() * 0.0
            out = {"loss_cos": zero, "loss_infonce": zero}
            return out

        z_all = torch.cat(matched_pred, dim=0)
        y_all = torch.cat(matched_tgt, dim=0) 

        losses = {}
        if self.embed_loss in ("cosine", "both") and self.lambda_cos > 0:
            loss_cos = 1.0 - (z_all * y_all).sum(dim=-1).mean()
            losses["loss_cos"] = loss_cos * self.lambda_cos

        if self.embed_loss in ("infonce", "both") and self.lambda_infonce > 0:
            loss_infonce = info_nce_loss(z_all, y_all, tau=self.tau)
            losses["loss_infonce"] = loss_infonce * self.lambda_infonce

        return losses


    def forward(self, outputs, epoch, targets):
        indices = self.matcher(outputs, epoch, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_embeddings(outputs, targets, indices))
        losses.update(self.loss_sentence_embedding(outputs, targets))
        losses.update(self.loss_attributes(outputs, targets))

        return losses

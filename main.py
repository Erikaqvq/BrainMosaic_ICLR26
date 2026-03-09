import json
import os
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from config_args import parse_args


def save_run_config(args):
    def _to_jsonable(v):
        if isinstance(v, Path):
            return str(v)
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    cfg = {k: _to_jsonable(v) for k, v in vars(args).items()}
    cfg["_cmd"] = " ".join(shlex.quote(s) for s in sys.argv)
    cfg["_time"] = datetime.now().isoformat(timespec="seconds")

    out_path = Path(args.output_dir) / "args.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def build_dataloaders(args, dataset_train, dataset_val):
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    return sampler_train, data_loader_train, data_loader_val


def main():
    args = parse_args()

    from datasets import build_dataset, load_token_bank
    from engine import evaluate, train_one_epoch
    from models import build_model

    utils.init_distributed_mode(args)

    if not args.output_dir:
        args.output_dir = f"outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    if utils.is_main_process():
        save_run_config(args)
        print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    token_bank = load_token_bank(
        args.token_path,
        dim=args.hidden_dim,
        normalize=bool(args.normalize_token_emb),
    )
    dataset_train = build_dataset(split="train", args=args, token_bank=token_bank)
    dataset_val = build_dataset(split="val", args=args, token_bank=token_bank)

    model, criterion, postprocessors = build_model(args, token_bank)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if utils.is_main_process():
        print(f"Number of parameters: {n_parameters}")

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    sampler_train, data_loader_train, data_loader_val = build_dataloaders(
        args, dataset_train, dataset_val
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if all(k in checkpoint for k in ("optimizer", "lr_scheduler", "epoch")):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        eval_out = evaluate(
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            data_loader=data_loader_val,
            device=device,
            epoch=None,
            save_dir=os.path.join(args.output_dir, "eval_only"),
        )
        stats = eval_out["stats"]
        if utils.is_main_process():
            print(f"Eval matching_acc={stats.get('matching_accuracy', 0.0):.4f}")
            print(f"Eval mean_cos={stats.get('mean_cosine', 0.0):.4f}")
        return

    if utils.is_main_process():
        print("Start training")

    best_acc = None
    best_ckpt_path = None
    best_epoch = None
    epoch_history = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        eval_out = evaluate(
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            data_loader=data_loader_val,
            device=device,
            epoch=epoch,
            best_acc=best_acc,
            best_ckpt_path=best_ckpt_path,
            save_dir=os.path.join(args.output_dir, "eval_embeddings"),
        )
        stats = eval_out["stats"]

        if eval_out.get("best_updated"):
            best_acc = float(stats.get("matching_accuracy", 0.0))
            best_ckpt_path = eval_out["best_ckpt_path"]
            best_epoch = eval_out.get("best_epoch", epoch)

        if utils.is_main_process():
            epoch_record = {
                "epoch": epoch,
                "train": {k: float(v) for k, v in train_stats.items()},
                "val": {k: float(v) for k, v in stats.items()},
            }
            epoch_history.append(epoch_record)
            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_stats.get('loss', 0.0):.4f} "
                f"val_loss={stats.get('loss', 0.0):.4f} "
                f"matching_acc={stats.get('matching_accuracy', 0.0):.4f} "
                f"mean_cos={stats.get('mean_cosine', 0.0):.4f}"
            )

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "epoch_history.json"), "w", encoding="utf-8") as f:
            json.dump(epoch_history, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_acc": best_acc,
                    "best_ckpt_path": best_ckpt_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()

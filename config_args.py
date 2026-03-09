import argparse
import json
from pathlib import Path

RECOMMENDED_DIM_MAP = {
    32: {"nheads": 2, "dim_ff": 128},
    64: {"nheads": 4, "dim_ff": 256},
    128: {"nheads": 8, "dim_ff": 512},
    256: {"nheads": 8, "dim_ff": 1024},
    512: {"nheads": 8, "dim_ff": 2048},
    1024: {"nheads": 16, "dim_ff": 4096},
}

DEFAULTS = {
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "epochs": 50,
    "lr_drop": 50,
    "clip_max_norm": 0.1,
    "token_path": None,
    "sent_emb_path": None,
    "segmentation_path": None,
    "eeg_path": None,
    "eeg_split_pattern": "{split}.pt",
    "eeg_scale": 1e6,
    "normalize_token_emb": False,
    "output_dir": "outputs/default",
    "device": "cuda",
    "seed": 42,
    "start_epoch": 0,
    "eval": False,
    "num_workers": 0,
    "batch_size": 32,
    "resume": "",
    "encoder": "moderntcn",
    "in_channels": 122,
    "tcn_blocks_per_stage": [2],
    "tcn_large_kernel_per_stage": [25],
    "tcn_small_kernel_per_stage": [5],
    "tcn_ffn_ratio": 2.0,
    "tcn_downsample_ratio": 1,
    "tcn_stem_dim": 122,
    "tcn_size": 64,
    "tcn_use_revin": False,
    "tcn_dropout": 0.0,
    "enc_layers": 3,
    "dec_layers": 6,
    "dim_feedforward": None,
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": None,
    "pre_norm": False,
    "num_queries": 8,
    "top_k": 5,
    "exist_threshold": 0.7,
    "cos_threshold": 0.7,
    "embed_loss": "both",
    "tau": 0.07,
    "lambda_infonce": 0.2,
    "lambda_cos": 1.0,
    "lambda_sent": 0.2,
    "lambda_cls": 1.0,
    "eos_coef": 0.3,
    "cost_class": 1.0,
    "cost_emb": 2.0,
    "lambda_sentence_mode": 0.2,
    "lambda_subjectivity": 0.2,
    "lambda_semantic_focus": 0.2,
    "sentence_mode_class_counts": None,
    "subjectivity_class_counts": None,
    "semantic_focus_class_counts": None,
    "slot_dropout_p": 0.2,
    "world_size": 1,
    "dist_url": "env://",
}

CONFIG_KEY_MAP = {
    "data.token_path": "token_path",
    "data.sent_emb_path": "sent_emb_path",
    "data.segmentation_path": "segmentation_path",
    "data.eeg_path": "eeg_path",
    "data.eeg_split_pattern": "eeg_split_pattern",
    "data.eeg_scale": "eeg_scale",
    "data.in_channels": "in_channels",
    "runtime.output_dir": "output_dir",
    "runtime.device": "device",
    "runtime.seed": "seed",
    "runtime.start_epoch": "start_epoch",
    "runtime.eval": "eval",
    "runtime.num_workers": "num_workers",
    "runtime.batch_size": "batch_size",
    "runtime.resume": "resume",
    "runtime.world_size": "world_size",
    "runtime.dist_url": "dist_url",
    "train.lr": "lr",
    "train.lr_backbone": "lr_backbone",
    "train.weight_decay": "weight_decay",
    "train.epochs": "epochs",
    "train.lr_drop": "lr_drop",
    "train.clip_max_norm": "clip_max_norm",
    "model.encoder": "encoder",
    "model.tcn_blocks_per_stage": "tcn_blocks_per_stage",
    "model.tcn_large_kernel_per_stage": "tcn_large_kernel_per_stage",
    "model.tcn_small_kernel_per_stage": "tcn_small_kernel_per_stage",
    "model.tcn_ffn_ratio": "tcn_ffn_ratio",
    "model.tcn_downsample_ratio": "tcn_downsample_ratio",
    "model.tcn_stem_dim": "tcn_stem_dim",
    "model.tcn_size": "tcn_size",
    "model.tcn_use_revin": "tcn_use_revin",
    "model.tcn_dropout": "tcn_dropout",
    "model.enc_layers": "enc_layers",
    "model.dec_layers": "dec_layers",
    "model.dim_feedforward": "dim_feedforward",
    "model.hidden_dim": "hidden_dim",
    "model.dropout": "dropout",
    "model.nheads": "nheads",
    "model.pre_norm": "pre_norm",
    "model.num_queries": "num_queries",
    "model.slot_dropout_p": "slot_dropout_p",
    "retrieval.top_k": "top_k",
    "retrieval.exist_threshold": "exist_threshold",
    "retrieval.cos_threshold": "cos_threshold",
    "loss.embed_loss": "embed_loss",
    "loss.tau": "tau",
    "loss.lambda_infonce": "lambda_infonce",
    "loss.lambda_cos": "lambda_cos",
    "loss.lambda_sent": "lambda_sent",
    "loss.lambda_cls": "lambda_cls",
    "loss.eos_coef": "eos_coef",
    "loss.lambda_sentence_mode": "lambda_sentence_mode",
    "loss.lambda_subjectivity": "lambda_subjectivity",
    "loss.lambda_semantic_focus": "lambda_semantic_focus",
    "loss.sentence_mode_class_counts": "sentence_mode_class_counts",
    "loss.subjectivity_class_counts": "subjectivity_class_counts",
    "loss.semantic_focus_class_counts": "semantic_focus_class_counts",
    "loss.lambda_te": "lambda_sentence_mode",
    "loss.lambda_oors": "lambda_subjectivity",
    "loss.lambda_su": "lambda_semantic_focus",
    "loss.te_class_counts": "sentence_mode_class_counts",
    "loss.oors_class_counts": "subjectivity_class_counts",
    "loss.su_class_counts": "semantic_focus_class_counts",
    "loss.cost_class": "cost_class",
    "loss.cost_emb": "cost_emb",
    "data.normalize_token_emb": "normalize_token_emb",
}


def _nested_get(mapping, dotted_key):
    cur = mapping
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_config_file(config_path):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    merged = {}
    for dotted_key, arg_key in CONFIG_KEY_MAP.items():
        value = _nested_get(raw, dotted_key)
        if value is not None:
            merged[arg_key] = value
    return merged


def apply_recommended_transformer_dims(args):
    cfg = RECOMMENDED_DIM_MAP.get(args.hidden_dim)
    if cfg is None:
        raise ValueError(f"hidden_dim {args.hidden_dim} not in {list(RECOMMENDED_DIM_MAP)}")
    if args.nheads is None:
        args.nheads = cfg["nheads"]
    if args.dim_feedforward is None:
        args.dim_feedforward = cfg["dim_ff"]
    if args.hidden_dim % args.nheads != 0:
        raise ValueError(
            f"hidden_dim({args.hidden_dim}) must be divisible by nheads({args.nheads})"
        )
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("BrainMosaic-SID")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")

    og = parser.add_argument_group("Optim")
    og.add_argument("--lr", default=DEFAULTS["lr"], type=float)
    og.add_argument("--lr_backbone", default=DEFAULTS["lr_backbone"], type=float)
    og.add_argument("--weight_decay", default=DEFAULTS["weight_decay"], type=float)
    og.add_argument("--epochs", default=DEFAULTS["epochs"], type=int)
    og.add_argument("--lr_drop", default=DEFAULTS["lr_drop"], type=int)
    og.add_argument("--clip_max_norm", default=DEFAULTS["clip_max_norm"], type=float)

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--token_path", default=DEFAULTS["token_path"], type=str)
    data_group.add_argument("--sent_emb_path", default=DEFAULTS["sent_emb_path"], type=str)
    data_group.add_argument("--segmentation_path", default=DEFAULTS["segmentation_path"], type=str)
    data_group.add_argument("--eeg_path", default=DEFAULTS["eeg_path"], type=str)
    data_group.add_argument("--eeg_split_pattern", default=DEFAULTS["eeg_split_pattern"], type=str)
    data_group.add_argument("--eeg_scale", default=DEFAULTS["eeg_scale"], type=float)
    data_group.add_argument("--in_channels", default=DEFAULTS["in_channels"], type=int)
    data_group.add_argument("--normalize_token_emb", action="store_true")
    data_group.add_argument("--output_dir", default=DEFAULTS["output_dir"], type=str)

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--device", default=DEFAULTS["device"], type=str)
    runtime_group.add_argument("--seed", default=DEFAULTS["seed"], type=int)
    runtime_group.add_argument("--start_epoch", default=DEFAULTS["start_epoch"], type=int)
    runtime_group.add_argument("--eval", action="store_true")
    runtime_group.add_argument("--num_workers", default=DEFAULTS["num_workers"], type=int)
    runtime_group.add_argument("--batch_size", default=DEFAULTS["batch_size"], type=int)
    runtime_group.add_argument("--resume", default=DEFAULTS["resume"], type=str)

    backbone_group = parser.add_argument_group("Backbone")
    backbone_group.add_argument(
        "--encoder",
        type=str,
        default=DEFAULTS["encoder"],
        choices=["moderntcn"],
    )
    backbone_group.add_argument("--tcn_blocks_per_stage", type=int, nargs="+", default=DEFAULTS["tcn_blocks_per_stage"])
    backbone_group.add_argument("--tcn_large_kernel_per_stage", type=int, nargs="+", default=DEFAULTS["tcn_large_kernel_per_stage"])
    backbone_group.add_argument("--tcn_small_kernel_per_stage", type=int, nargs="+", default=DEFAULTS["tcn_small_kernel_per_stage"])
    backbone_group.add_argument("--tcn_ffn_ratio", type=float, default=DEFAULTS["tcn_ffn_ratio"])
    backbone_group.add_argument("--tcn_downsample_ratio", type=int, default=DEFAULTS["tcn_downsample_ratio"])
    backbone_group.add_argument("--tcn_stem_dim", type=int, default=DEFAULTS["tcn_stem_dim"])
    backbone_group.add_argument("--tcn_size", type=int, default=DEFAULTS["tcn_size"])
    backbone_group.add_argument("--tcn_use_revin", action="store_true")
    backbone_group.add_argument("--tcn_dropout", type=float, default=DEFAULTS["tcn_dropout"])

    transformer_group = parser.add_argument_group("Transformer")
    transformer_group.add_argument("--enc_layers", default=DEFAULTS["enc_layers"], type=int)
    transformer_group.add_argument("--dec_layers", default=DEFAULTS["dec_layers"], type=int)
    transformer_group.add_argument("--dim_feedforward", type=int, default=DEFAULTS["dim_feedforward"])
    transformer_group.add_argument("--hidden_dim", type=int, default=DEFAULTS["hidden_dim"])
    transformer_group.add_argument("--dropout", default=DEFAULTS["dropout"], type=float)
    transformer_group.add_argument("--nheads", type=int, default=DEFAULTS["nheads"])
    transformer_group.add_argument("--pre_norm", action="store_true")

    matching_group = parser.add_argument_group("DETR / Matching")
    matching_group.add_argument("--num_queries", default=DEFAULTS["num_queries"], type=int)
    matching_group.add_argument("--top_k", default=DEFAULTS["top_k"], type=int)
    matching_group.add_argument("--exist_threshold", default=DEFAULTS["exist_threshold"], type=float)
    matching_group.add_argument("--cos_threshold", default=DEFAULTS["cos_threshold"], type=float)

    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--embed_loss", choices=["infonce", "cosine", "both"], default=DEFAULTS["embed_loss"])
    loss_group.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    loss_group.add_argument("--lambda_infonce", type=float, default=DEFAULTS["lambda_infonce"])
    loss_group.add_argument("--lambda_cos", type=float, default=DEFAULTS["lambda_cos"])
    loss_group.add_argument("--lambda_sent", type=float, default=DEFAULTS["lambda_sent"])
    loss_group.add_argument("--lambda_cls", type=float, default=DEFAULTS["lambda_cls"])
    loss_group.add_argument("--eos_coef", default=DEFAULTS["eos_coef"], type=float)
    loss_group.add_argument("--lambda_sentence_mode", type=float, default=DEFAULTS["lambda_sentence_mode"])
    loss_group.add_argument("--lambda_subjectivity", type=float, default=DEFAULTS["lambda_subjectivity"])
    loss_group.add_argument("--lambda_semantic_focus", type=float, default=DEFAULTS["lambda_semantic_focus"])
    loss_group.add_argument("--sentence_mode_class_counts", type=float, nargs="+", default=DEFAULTS["sentence_mode_class_counts"])
    loss_group.add_argument("--subjectivity_class_counts", type=float, nargs="+", default=DEFAULTS["subjectivity_class_counts"])
    loss_group.add_argument("--semantic_focus_class_counts", type=float, nargs="+", default=DEFAULTS["semantic_focus_class_counts"])
    loss_group.add_argument("--cost_class", type=float, default=DEFAULTS["cost_class"])
    loss_group.add_argument("--cost_emb", type=float, default=DEFAULTS["cost_emb"])
    loss_group.add_argument("--slot_dropout_p", type=float, default=DEFAULTS["slot_dropout_p"])

    distributed_group = parser.add_argument_group("Distributed")
    distributed_group.add_argument("--world_size", default=DEFAULTS["world_size"], type=int)
    distributed_group.add_argument("--dist_url", default=DEFAULTS["dist_url"], type=str)
    return parser


def parse_args():
    parser = get_args_parser()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = DEFAULTS.copy()
    config_defaults.update(_load_config_file(pre_args.config))
    parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    args = apply_recommended_transformer_dims(args)

    required_paths = ["token_path", "sent_emb_path", "eeg_path"]
    missing = [k for k in required_paths if getattr(args, k) in (None, "")]
    if missing:
        raise ValueError(
            f"Missing required path fields: {missing}. Set them in --config or CLI arguments."
        )
    return args

import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SUBJECT_ID = {"I/we": 0, "you": 1, "others": 2, "thing": 3, "event": 4}
OS_ID = {"objective": 0, "subjective": 1}
TEXTUAL_ID = {"statement": 0, "question": 1, "negative": 2, "imperative": 3}


def load_token_bank(token_path, dim, normalize=True):
    cluster_emb_path = os.path.join(token_path, "embeddings.pt")
    cluster_map_path = os.path.join(token_path, "map.json")
    cluster_info_path = os.path.join(token_path, "info.json")

    data = torch.load(cluster_emb_path, map_location="cpu")
    cluster_emb = data["embeddings"].to(torch.float32)
    if normalize:
        cluster_emb = F.normalize(cluster_emb, dim=1)

    if cluster_emb.size(1) < dim:
        raise ValueError(f"cluster_emb dim={cluster_emb.size(1)} < requested dim={dim}")
    if cluster_emb.size(1) > dim:
        cluster_emb = cluster_emb[:, :dim]

    with open(cluster_map_path, "r", encoding="utf-8") as f:
        word2cid = json.load(f)

    with open(cluster_info_path, "r", encoding="utf-8") as f:
        cluster_info = json.load(f)
    if len(cluster_info) != cluster_emb.size(0):
        raise ValueError(
            f"cluster_info length {len(cluster_info)} does not match {cluster_emb.size(0)}"
        )
    cid2members_all = {
        item["cluster2_id"]: list(item.get("members", [])) for item in cluster_info
    }

    return {
        "emb": cluster_emb,
        "map": word2cid,
        "cluster_members": cid2members_all,
        "dim": dim,
    }


def load_sentence_embedding_bank(path, dim=None):
    data = torch.load(path, map_location="cpu")
    sentences = data["sentences"]
    embeddings = data["embeddings"].float()

    if dim is not None and embeddings.size(1) > dim:
        embeddings = embeddings[:, :dim]
    embeddings = F.normalize(embeddings, dim=-1)

    if len(sentences) != embeddings.shape[0]:
        raise ValueError("Mismatch between sentence list and sentence embeddings")
    return {str(s).strip(): e for s, e in zip(sentences, embeddings)}


def _load_tabular(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return pd.DataFrame(raw)
    raise ValueError(f"Unsupported segmentation format: {ext}")


def _to_int_label(value, mapping):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return mapping.get(text)


def load_segmentation_lookup(path):
    if not path:
        return {}

    seg_df = _load_tabular(path)
    if "sentence" not in seg_df.columns:
        raise ValueError("Segmentation file must include a 'sentence' column")

    seg_df = seg_df.dropna(subset=["sentence"])
    seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

    token_cols = [
        c
        for c in seg_df.columns
        if str(c).startswith("words") or re.fullmatch(r"word\d+", str(c))
    ]

    def _pick_value(row, keys):
        for key in keys:
            if key in row and pd.notna(row[key]):
                return row[key]
        return None

    lookup = {}
    for _, row in seg_df.iterrows():
        sentence = str(row["sentence"]).strip()
        if not sentence:
            continue

        tokens = []
        if "tokens" in row and isinstance(row["tokens"], list):
            tokens = [str(x).strip() for x in row["tokens"] if str(x).strip()]
        else:
            for c in token_cols:
                val = row.get(c, None)
                if pd.notna(val):
                    sval = str(val).strip()
                    if sval and sval.lower() != "nan":
                        tokens.append(sval)

        sentence_mode_raw = _pick_value(row, ["sentence_mode", "te", "textual"])
        subjectivity_raw = _pick_value(row, ["subjectivity", "oors", "OS"])
        semantic_focus_raw = _pick_value(row, ["semantic_focus", "su", "subject"])

        lookup[sentence] = {
            "tokens": tokens,
            "sentence_mode": _to_int_label(sentence_mode_raw, TEXTUAL_ID),
            "subjectivity": _to_int_label(subjectivity_raw, OS_ID),
            "semantic_focus": _to_int_label(semantic_focus_raw, SUBJECT_ID),
        }

    return lookup


def _coerce_eeg_tensor(raw, in_channels, eeg_scale):
    if isinstance(raw, torch.Tensor):
        eeg = raw.detach().cpu().to(torch.float32)
    else:
        eeg = torch.as_tensor(raw, dtype=torch.float32)

    if eeg.ndim != 2:
        raise ValueError(f"EEG sample must be 2D, got shape={tuple(eeg.shape)}")

    c, t = eeg.shape
    if c == in_channels:
        eeg_channels_first = eeg
    elif t == in_channels:
        eeg_channels_first = eeg.transpose(0, 1)
    else:
        raise ValueError(
            f"EEG shape {tuple(eeg.shape)} cannot align with in_channels={in_channels}"
        )

    if eeg_scale is not None:
        eeg_channels_first = eeg_channels_first * float(eeg_scale)

    return eeg_channels_first


def _load_split_records(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        if not isinstance(data, list):
            raise ValueError("PT split file must be a list[dict] or {'samples': list[dict]}")
        return data

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        if not isinstance(data, list):
            raise ValueError("JSON split file must be a list[dict] or {'samples': list[dict]}")
        return data

    raise ValueError(f"Unsupported split file extension: {ext}")


class UnifiedEEGDataset(Dataset):
    def __init__(self, args, split="train", token_bank=None):
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported split: {split}")
        if token_bank is None:
            raise ValueError("token_bank is required")

        self.split = split
        self.token_emb = token_bank["emb"]
        self.token_map = token_bank["map"]
        self.dim = token_bank["dim"]

        self.sent_emb_dict = load_sentence_embedding_bank(args.sent_emb_path, dim=self.dim)
        self.seg_lookup = load_segmentation_lookup(getattr(args, "segmentation_path", None))

        split_file = os.path.join(args.eeg_path, args.eeg_split_pattern.format(split=split))
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Unified split file not found: {split_file}")

        raw_records = _load_split_records(split_file)
        self.data = []

        for rec in raw_records:
            if not isinstance(rec, dict):
                continue

            sentence = str(rec.get("sentence", "")).strip()
            if not sentence or sentence not in self.sent_emb_dict:
                continue

            seg_meta = self.seg_lookup.get(sentence, {})
            raw_tokens = rec.get("words", seg_meta.get("tokens", [])) or []
            tokens = [str(w).strip() for w in raw_tokens if str(w).strip()]
            kept_tokens = [w for w in tokens if w in self.token_map]
            if not kept_tokens:
                continue

            eeg_raw = rec.get("eeg", None)
            if eeg_raw is None:
                continue

            eeg_channels_first = _coerce_eeg_tensor(
                eeg_raw,
                in_channels=args.in_channels,
                eeg_scale=args.eeg_scale,
            )

            sentence_mode = rec.get(
                "sentence_mode", rec.get("te", seg_meta.get("sentence_mode", 0))
            )
            subjectivity = rec.get(
                "subjectivity", rec.get("oors", seg_meta.get("subjectivity", 0))
            )
            semantic_focus = rec.get(
                "semantic_focus", rec.get("su", seg_meta.get("semantic_focus", 0))
            )

            sentence_mode = _to_int_label(sentence_mode, TEXTUAL_ID)
            subjectivity = _to_int_label(subjectivity, OS_ID)
            semantic_focus = _to_int_label(semantic_focus, SUBJECT_ID)
            sentence_mode = 0 if sentence_mode is None else int(sentence_mode)
            subjectivity = 0 if subjectivity is None else int(subjectivity)
            semantic_focus = 0 if semantic_focus is None else int(semantic_focus)

            cid_list = [self.token_map[w] for w in kept_tokens]
            cluster_embs = self.token_emb[torch.tensor(cid_list, dtype=torch.long)]

            self.data.append(
                (
                    eeg_channels_first,
                    {
                        "sentence": sentence,
                        "sentence_embedding": self.sent_emb_dict[sentence],
                        "words": kept_tokens,
                        "sentence_mode": sentence_mode,
                        "semantic_focus": semantic_focus,
                        "subjectivity": subjectivity,
                        "word_embeddings": cluster_embs,
                        "word_cluster_ids": cid_list,
                    },
                )
            )

        if not self.data:
            raise ValueError(
                f"No valid samples loaded for split '{split}' from {split_file}. "
                "Check unified data schema and text assets alignment."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

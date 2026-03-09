import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if p.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = 128,
    truncate_dim: int = 256,
):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        hidden = model(**tokens).last_hidden_state
        pooled = mean_pool(hidden, tokens["attention_mask"])
        if truncate_dim is not None and truncate_dim > 0:
            pooled = pooled[:, :truncate_dim]
        pooled = F.normalize(pooled, dim=-1)
        outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, truncate_dim)


def main():
    parser = argparse.ArgumentParser("Generate public text embeddings for BrainMosaic")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name_or_path"]
    device = cfg["runtime"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["runtime"].get("batch_size", 32))
    max_length = int(cfg["runtime"].get("max_length", 128))
    truncate_dim = int(cfg["runtime"].get("truncate_dim", 256))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(device)

    sentence_df = read_table(cfg["inputs"]["sentences_file"])
    sentence_col = cfg["inputs"].get("sentence_col", "sentence")
    sentences = sentence_df[sentence_col].dropna().astype(str).str.strip().tolist()
    sentence_emb = encode_texts(
        sentences, tokenizer, model, batch_size=batch_size, max_length=max_length, truncate_dim=truncate_dim
    )
    torch.save(
        {"sentences": sentences, "embeddings": sentence_emb},
        str(out_dir / "sentence_embeddings.pt"),
    )

    token_df = read_table(cfg["inputs"]["tokens_file"])
    token_key_col = cfg["inputs"].get("token_key_col", "key")
    token_exp_col = cfg["inputs"].get("token_explanation_col", "explanation")
    use_expansion = bool(cfg["inputs"].get("use_expansion_text", True))

    keys = token_df[token_key_col].dropna().astype(str).str.strip().tolist()
    key_to_text = {}
    if use_expansion and token_exp_col in token_df.columns:
        for _, row in token_df.iterrows():
            k = str(row.get(token_key_col, "")).strip()
            e = str(row.get(token_exp_col, "")).strip()
            if k:
                key_to_text[k] = e if e else k
    else:
        key_to_text = {k: k for k in keys}

    keys = [k for k in keys if k in key_to_text]
    texts = [key_to_text[k] for k in keys]
    word_emb = encode_texts(
        texts, tokenizer, model, batch_size=batch_size, max_length=max_length, truncate_dim=truncate_dim
    )
    torch.save(
        {"keys": keys, "texts": texts, "embeddings": word_emb},
        str(out_dir / "word_embeddings.pt"),
    )

    print(f"[OK] sentence_embeddings.pt: {len(sentences)} x {sentence_emb.shape[-1]}")
    print(f"[OK] word_embeddings.pt: {len(keys)} x {word_emb.shape[-1]}")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_clusters_complete_linkage(norm_emb: torch.Tensor, sim_threshold: float):
    sim = norm_emb @ norm_emb.t()
    sim.fill_diagonal_(1.0)
    n = sim.size(0)
    visited = torch.zeros(n, dtype=torch.bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        current = [i]
        visited[i] = True
        while True:
            cand = torch.where(~visited)[0]
            if cand.numel() == 0:
                break
            sub = sim.index_select(0, cand).index_select(1, torch.tensor(current))
            min_sim = sub.min(dim=1).values
            keep = cand[min_sim >= sim_threshold]
            if keep.numel() == 0:
                break
            current.extend(keep.tolist())
            visited[keep] = True
        clusters.append(current)
    return clusters


def main():
    parser = argparse.ArgumentParser("Build public token bank for BrainMosaic")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    in_file = Path(cfg["input"]["word_embeddings_pt"])
    out_dir = Path(cfg["output"]["token_bank_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(in_file, map_location="cpu")
    keys = [str(k).strip() for k in data["keys"]]
    emb = data["embeddings"].float()

    truncate_dim = int(cfg.get("truncate_dim", emb.shape[1]))
    if truncate_dim > 0 and emb.shape[1] > truncate_dim:
        emb = emb[:, :truncate_dim]
    emb = F.normalize(emb, dim=1)

    sim_threshold = float(cfg.get("cluster_sim_threshold", 0.78))
    clusters = build_clusters_complete_linkage(emb, sim_threshold=sim_threshold)

    cluster_embeddings = []
    cluster_info = []
    word2cid = {}
    for cid, members_idx in enumerate(clusters):
        members = [keys[i] for i in members_idx]
        cemb = F.normalize(emb[members_idx].mean(dim=0), dim=0)
        cluster_embeddings.append(cemb.unsqueeze(0))
        for w in members:
            word2cid[w] = cid
        cluster_info.append(
            {
                "cluster2_id": cid,
                "size": len(members),
                "members": members,
            }
        )

    cluster_embeddings = torch.cat(cluster_embeddings, dim=0) if cluster_embeddings else torch.empty(0, emb.shape[1])
    torch.save({"embeddings": cluster_embeddings}, str(out_dir / "embeddings.pt"))
    with open(out_dir / "map.json", "w", encoding="utf-8") as f:
        json.dump(word2cid, f, ensure_ascii=False, indent=2)
    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)

    print(f"[OK] clusters={len(clusters)}")
    print(f"[OUT] {out_dir / 'embeddings.pt'}")
    print(f"[OUT] {out_dir / 'map.json'}")
    print(f"[OUT] {out_dir / 'info.json'}")


if __name__ == "__main__":
    main()

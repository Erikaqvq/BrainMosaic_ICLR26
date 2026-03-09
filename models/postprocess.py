import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import pdb

class PostProcess:
    def __init__(self, token_bank, top_k=5,
                 exist_threshold=0.5, cos_threshold=0.7):

        self.cluster_emb = token_bank["emb"] 
        self.word2cid     = token_bank["map"]
        self.cid2members  = token_bank["cluster_members"]

        self.top_k1 = int(top_k)
        self.top_k2 = 20
        self.exist_threshold = float(exist_threshold)
        self.cos_threshold   = float(cos_threshold)


    @torch.no_grad()
    def forward(self, outputs, epoch, targets=None):
        assert "pred_embeddings" in outputs, "no pred_embeddings in outputs!"
        preds = outputs["pred_embeddings"]
        preds = F.normalize(preds, dim=-1)
        
        vocab = F.normalize(self.cluster_emb.to(preds.device), dim=-1)

        has_logits = ("pred_logits" in outputs)
        content_probability_all = outputs["pred_logits"].softmax(-1)[..., 1] if has_logits else None

        sentence_mode_probs = (
            F.softmax(outputs["sentence_mode"], dim=-1) if "sentence_mode" in outputs else None
        )
        subjectivity_probs = (
            F.softmax(outputs["subjectivity"], dim=-1) if "subjectivity" in outputs else None
        )
        semantic_focus_probs = (
            F.softmax(outputs["semantic_focus"], dim=-1) if "semantic_focus" in outputs else None
        )

        B, N, D = preds.shape
        C = vocab.size(0)
        results = []

        for b in range(B):
                                       
            sim = preds[b] @ vocab.t()
            top1_sim, _ = sim.max(dim=1)       

            if has_logits:
                keep_mask = (content_probability_all[b] > self.exist_threshold) & (top1_sim > self.cos_threshold)
            else:
                keep_mask = (top1_sim > self.cos_threshold)

            keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)       
            M = int(keep_idx.numel())

            if M == 0:
                results.append({
                    "content_probabilities": [] if has_logits else None,
                    "topk_words": [],
                    "topk_scores": [],
                    "keep_idx": [],
                    "sentence_mode_probs": (
                        sentence_mode_probs[b].detach().cpu().tolist()
                        if sentence_mode_probs is not None
                        else None
                    ),
                    "subjectivity_probs": (
                        subjectivity_probs[b].detach().cpu().tolist()
                        if subjectivity_probs is not None
                        else None
                    ),
                    "semantic_focus_probs": (
                        semantic_focus_probs[b].detach().cpu().tolist()
                        if semantic_focus_probs is not None
                        else None
                    ),
                })
                continue

            sim_keep = sim[keep_idx]
            k2 = min(self.top_k2, sim_keep.size(1))
            sims_k2, cids_k2 = torch.topk(sim_keep, k=k2, dim=1)
            best_for_cid = {} 
            for q in range(M):
                for r in range(k2):
                    cid = int(cids_k2[q, r])
                    score = float(sims_k2[q, r])
                    cur = best_for_cid.get(cid)
                    if (cur is None) or (score > cur[1]):
                        best_for_cid[cid] = (q, score, r)

            per_query_picks = [[] for _ in range(M)]
            for cid, (q, score, r) in best_for_cid.items():
                per_query_picks[q].append((r, cid, score))

            selected_cids  = []
            selected_sims  = []
            for q in range(M):
                per_query_picks[q].sort(key=lambda x: x[0])
                take = per_query_picks[q][:self.top_k1]
                selected_cids.append([cid   for r, cid, s in take])
                selected_sims.append([float(s) for r, cid, s in take])

            topk_words_kept = []
            for q in range(M):
                words_lists = [self.cid2members.get(cid, []) for cid in selected_cids[q]]
                topk_words_kept.append(words_lists)

            result_item = {
                "content_probabilities": (
                    content_probability_all[b][keep_idx].detach().cpu().tolist()
                    if has_logits
                    else None
                ),
                "topk_words": topk_words_kept, 
                "topk_scores": selected_sims, 
                "keep_idx": keep_idx.detach().cpu().tolist(),
                "sentence_mode_probs": (
                    sentence_mode_probs[b].detach().cpu().tolist()
                    if sentence_mode_probs is not None
                    else None
                ),
                "subjectivity_probs": (
                    subjectivity_probs[b].detach().cpu().tolist()
                    if subjectivity_probs is not None
                    else None
                ),
                "semantic_focus_probs": (
                    semantic_focus_probs[b].detach().cpu().tolist()
                    if semantic_focus_probs is not None
                    else None
                ),
            }
            results.append(result_item)
        
        return results

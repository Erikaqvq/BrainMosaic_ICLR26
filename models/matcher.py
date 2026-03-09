import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcherEmbedding(torch.nn.Module):
    def __init__(self, cost_class=0.5, cost_emb=1.0,):
        super().__init__()
        self.cost_class = cost_class
        self.cost_emb = cost_emb

    @torch.no_grad()
    def forward(self, outputs, epoch, targets):
        B, N, _ = outputs["pred_embeddings"].shape
        out_prob = (outputs["pred_logits"].softmax(-1)) 

        if out_prob.size(-1) == 2:
            content_probability = out_prob[..., 1]
        else:
            content_probability = out_prob.squeeze(-1)

        pred_z = F.normalize(outputs["pred_embeddings"], dim=-1)

        indices = []
        for b in range(B):
            tgt_emb = targets[b]["word_embeddings"]
            if tgt_emb.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
                continue

            tgt_emb = F.normalize(tgt_emb, dim=-1)
            M = tgt_emb.size(0)

            cost_cls = -content_probability[b].unsqueeze(1).expand(N, M)

            cost_emb = 1.0 - pred_z[b] @ tgt_emb.t()

            C = self.cost_class * cost_cls + self.cost_emb * cost_emb         

            C[0, :] = 1e9

            row_ind, col_ind = linear_sum_assignment(C.cpu())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices

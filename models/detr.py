import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import json

from .backbone import build_backbone
from .matcher import HungarianMatcherEmbedding
from .transformer import build_transformer
from .criterion import SetCriterion
from .postprocess import PostProcess

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_queries, d_model, slot_dropout_p=0.0):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.query_embed.weight, mean=0, std=0.2)
        self.input_proj = nn.Linear(backbone.num_channels, d_model)
        self.backbone = backbone
        self.transformer = transformer

        self.class_head = nn.Linear(d_model, 2)
        self.emb_head = nn.Linear(d_model, d_model)
        self.sentence_mode_head = nn.Linear(d_model, 4)
        self.subjectivity_head = nn.Linear(d_model, 2)
        self.semantic_focus_head = nn.Linear(d_model, 5)

        self.num_queries = num_queries
        self.slot_dropout_p = float(slot_dropout_p) 

    def forward(self, samples):
        features, pos = self.backbone(samples)
        src = features
        mask = torch.zeros(src.shape[:2], dtype=torch.bool, device=src.device)
        qpos = self.query_embed.weight
        if self.training and self.slot_dropout_p > 0.0:
            keep = (torch.rand(qpos.size(0), device=qpos.device) > self.slot_dropout_p)
            keep[0] = True
            qpos = qpos * keep.float().unsqueeze(1)

        hs = self.transformer(self.input_proj(src), mask, qpos, pos)[0]

        logits_all = self.class_head(hs)
        emb_all    = self.emb_head(hs)

        sen_emb = emb_all[-1,:,0,:]
        sentence_mode = self.sentence_mode_head(sen_emb)
        subjectivity = self.subjectivity_head(sen_emb)
        semantic_focus = self.semantic_focus_head(sen_emb)

        out = {
            'pred_logits': logits_all[-1],
            'pred_embeddings': emb_all[-1],
            'sentence_mode': sentence_mode,
            'subjectivity': subjectivity,
            'semantic_focus': semantic_focus
        }
        return out


def build(args, token_bank):
    device = torch.device(args.device)
    
    model = DETR(
        backbone=build_backbone(args),
        transformer=build_transformer(args),
        num_queries=args.num_queries,
        d_model=args.hidden_dim,
        slot_dropout_p=getattr(args, "slot_dropout_p", 0.0) 
    )
    matcher = HungarianMatcherEmbedding(
        cost_class=getattr(args, 'cost_class', 1.0),
        cost_emb=getattr(args, 'cost_emb', 2.0),
    )
    post = PostProcess(
        token_bank=token_bank,
        top_k=args.top_k,
        exist_threshold=args.exist_threshold, 
        cos_threshold=args.cos_threshold
    )

    weight_dict = {
        "loss_cls": getattr(args, 'lambda_cls', 1.0),
        "loss_infonce": getattr(args, 'lambda_infonce', 1.0) if args.embed_loss in ('infonce','both') else 0.0,
        "loss_cos": getattr(args, 'lambda_cos', 1.0) if args.embed_loss in ('cosine','both') else 0.0,
    }

    criterion = SetCriterion(
        matcher=matcher,
        num_classes=2, 
        eos_coef=getattr(args, 'eos_coef', 0.1),
        lambda_cls=getattr(args, 'lambda_cls', 0.5),
        embed_loss=getattr(args, 'embed_loss', 'infonce'),
        tau=getattr(args, 'tau', 0.07),
        lambda_infonce=getattr(args, 'lambda_infonce', 1.0),
        lambda_cos=getattr(args, 'lambda_cos', 1.0),
        lambda_sentence_mode=getattr(args, 'lambda_sentence_mode', 0.5),
        lambda_subjectivity=getattr(args, 'lambda_subjectivity', 0.5),
        lambda_semantic_focus=getattr(args, 'lambda_semantic_focus', 0.5),
        sentence_mode_class_counts=getattr(args, "sentence_mode_class_counts", None),
        subjectivity_class_counts=getattr(args, "subjectivity_class_counts", None),
        semantic_focus_class_counts=getattr(args, "semantic_focus_class_counts", None),
    ).to(device)

    return model.to(device), criterion, {"topk": post, "weight_dict": weight_dict}

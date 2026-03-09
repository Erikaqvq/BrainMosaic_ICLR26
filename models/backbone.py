import torch
import torch.nn.functional as F
import torch
from torch import Tensor
from collections import defaultdict, deque
from packaging import version
from typing import Optional

from .ModernTCN import ModernTCNEncoder

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _ensure_channels_first(x, expected_channels:int):
                               
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B,*,*], got {x.shape}")
    B, A, B_or_C = x.shape[0], x.shape[1], x.shape[2]
    if A == expected_channels:
        return x
    if B_or_C == expected_channels:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"Input has wrong shape {x.shape}, expected channels={expected_channels} on dim 1 or 2.")


class Joiner(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        features = self.encoder(x) 
        return features, None


def build_backbone(args):
    encoder_name = getattr(args, "encoder", "moderntcn").lower()
    if encoder_name != "moderntcn":
        raise NotImplementedError(
            f"Encoder '{encoder_name}' is not implemented in this public release."
        )

    in_ch = getattr(args, "in_channels", 122)
    d_model = args.hidden_dim

    tcn_size = getattr(args, "tcn_size", 64)
    tcn_blocks_per_stage = getattr(args, "tcn_blocks_per_stage", [2])
    tcn_large_kernel_per_stage = getattr(args, "tcn_large_kernel_per_stage", [25])
    tcn_small_kernel_per_stage = getattr(args, "tcn_small_kernel_per_stage", [5])
    tcn_ffn_ratio = getattr(args, "tcn_ffn_ratio", 2.0)
    tcn_downsample_ratio = getattr(args, "tcn_downsample_ratio", 1)
    tcn_stem_dim = getattr(args, "tcn_stem_dim", in_ch)
    tcn_use_revin = bool(getattr(args, "tcn_use_revin", False))
    tcn_dropout = float(getattr(args, "tcn_dropout", 0.0))

    encoder = ModernTCNEncoder(
        in_channels=in_ch,
        d_model=tcn_size,
        patch_size=25, patch_stride=25,
        num_blocks_per_stage=tcn_blocks_per_stage,
        large_kernel_per_stage=tcn_large_kernel_per_stage,
        small_kernel_per_stage=tcn_small_kernel_per_stage,
        ffn_ratio=tcn_ffn_ratio,
        downsample_ratio=tcn_downsample_ratio,
        stem_dimension=tcn_stem_dim,
        use_revin=tcn_use_revin,
        dropout_backbone=tcn_dropout,
        embed_dim=d_model,
        use_learnable_var_mix=True,
        use_attn_pool=True,
        l2_normalize=True
        )
    
    model = Joiner(encoder)
    
    model.num_channels = tcn_size
    return model

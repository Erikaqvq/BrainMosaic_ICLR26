import einops
import torch
from torch import nn
import torch.nn.functional as F


class FixedLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D).permute(0, 1, 3, 2)
        return x

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'small_kernel cannot be larger than kernel_size'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):
                             
        D_out, D_in, _ = x.shape
        pad_left = x.new_full((D_out, D_in, pad_length_left), fill_value=pad_values)
        pad_right = x.new_full((D_out, D_in, pad_length_right), fill_value=pad_values)
        x = torch.cat([pad_left, x, pad_right], dim=-1)           
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(
                small_k, (self.kernel_size - self.small_kernel) // 2,
                (self.kernel_size - self.small_kernel) // 2, 0
            )
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        inp = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)

        x = inp + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars,
                 small_kernel_merged=False, drop=0.1):
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        self.blocks = nn.ModuleList([
            Block(large_size, small_size, dmodel, d_ffn, nvars, small_kernel_merged, drop)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-5)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ModernTCN(nn.Module):
    def __init__(self, patch_size, patch_stride, stem_dimension, downsample_ratio, ffn_ratio, num_blocks, large_size,
                 small_size, dims, nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1,
                 use_multi_scale=True, revin=True, affine=True, subtract_last=False, seq_len=512, c_in=7,
                 individual=False, target_window=96, class_drop=0., class_num=10):
        super(ModernTCN, self).__init__()
        self.class_drop = class_drop
        self.class_num = class_num

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(stem_dimension, affine=affine, subtract_last=subtract_last)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        self.stem = ConvBNReLU(c_in, stem_dimension, kernel_size=3, stride=1)

        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                          dmodel=dims[stage_idx], nvars=stem_dimension, small_kernel_merged=small_kernel_merged,
                          drop=backbone_dropout)
            self.stages.append(layer)

        self.act_class = F.gelu
        self.class_dropout = nn.Dropout(self.class_drop)
        d_model = dims[self.num_stage - 1]
        self.head_class = nn.Linear(stem_dimension * d_model, self.class_num)

    def forward_feature(self, x, aux_labels=None):
        B, M, L = x.shape
        x = x.unsqueeze(-2)                

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)

            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.forward_feature(x, aux_labels=None)
        x = self.act_class(x)
        x = self.class_dropout(x)
        x = einops.reduce(x, 'b c d s -> b c d', reduction='mean')
        x = einops.rearrange(x, 'b c d -> b (c d)')
        x = self.head_class(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class ModernTCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 122,
        d_model: int = 64,
        patch_size: int = 25,
        patch_stride: int = None,
        num_blocks_per_stage = [2],
        large_kernel_per_stage = [25],
        small_kernel_per_stage = [5],
        ffn_ratio: int = 2,
        downsample_ratio: int = 1,
        stem_dimension: int = None,
        use_revin: bool = False,
        dropout_backbone: float = 0.0,
        embed_dim: int = 512,
        use_learnable_var_mix: bool = True,
        use_attn_pool: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        if patch_stride is None:
            patch_stride = patch_size
        if stem_dimension is None:
            stem_dimension = in_channels

        dims = [d_model for _ in num_blocks_per_stage]

        self.backbone = ModernTCN(
            patch_size=patch_size,
            patch_stride=patch_stride,
            stem_dimension=stem_dimension,
            downsample_ratio=downsample_ratio,
            ffn_ratio=ffn_ratio,
            num_blocks=num_blocks_per_stage,
            large_size=large_kernel_per_stage,
            small_size=small_kernel_per_stage,
            dims=dims,
            nvars=stem_dimension,
            small_kernel_merged=False,
            backbone_dropout=dropout_backbone,
            head_dropout=0.0,
            use_multi_scale=False,
            revin=use_revin,
            affine=True,
            subtract_last=False,
            seq_len=512,
            c_in=in_channels,
            individual=False,
            target_window=0,
            class_drop=0.0,
            class_num=1,
        )

        self.d_model = dims[-1]
        self.patch_stride = patch_stride
        self.use_learnable_var_mix = use_learnable_var_mix
        self.use_attn_pool = use_attn_pool
        self.l2_normalize = l2_normalize

        if self.use_learnable_var_mix:
            self.mix_vars = nn.Conv2d(
                in_channels=stem_dimension, out_channels=1, kernel_size=1, stride=1, bias=False
            )
        else:
            self.mix_vars = None

        if self.use_attn_pool:
            self.attn_query = nn.Parameter(torch.randn(self.d_model))
        else:
            self.register_parameter('attn_query', None)

        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, embed_dim, bias=True),
        )

    @torch.no_grad()
    def _feature_only(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            C_expected = self.backbone.stem.conv.in_channels 
            if x.shape[1] != C_expected and x.shape[2] == C_expected:
                x = x.transpose(1, 2).contiguous()
        x = self.backbone.stem(x)
        x = self.backbone.forward_feature(x)
        return x

    def _mix_variables(self, feats: torch.Tensor) -> torch.Tensor:
        if self.mix_vars is None:
            return feats.mean(dim=1, keepdim=True)
        else:
            return self.mix_vars(feats)                

    def _temporal_pool(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.attn_query is None:
            return x_seq.mean(dim=1)
        q = self.attn_query / (self.attn_query.norm(p=2) + 1e-6)
        attn = torch.matmul(x_seq, q) 
        attn = torch.softmax(attn, dim=1).unsqueeze(-1) 
        pooled = (x_seq * attn).sum(dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._feature_only(x) 
        x_mixed = self._mix_variables(feats)
        x_seq = einops.rearrange(x_mixed, 'b 1 d n -> b n d')
        return x_seq 

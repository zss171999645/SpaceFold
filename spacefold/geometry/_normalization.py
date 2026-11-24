"""State-compatible channel LayerNorm for narrow octree features."""

import torch
import torch.nn.functional as F
from torch import nn


def _layer_norm_fp32(features, weight, bias, eps):
    x = features.float()
    if not x.is_cuda or not x.numel() or torch.is_grad_enabled():
        return F.layer_norm(x, weight.shape, weight, bias, eps)

    from ._kernel import layer_norm_tile

    x = x.contiguous()
    y = torch.empty_like(x)
    channels = x.shape[-1]
    rows = x.numel() // channels
    tile = 128
    layer_norm_tile[((rows + tile - 1) // tile,)](
        x, weight, bias, y, rows, channels, eps, tile, num_warps=4, enable_fp_fusion=False
    )
    return y


class _DenseNorm(nn.LayerNorm):
    def forward(self, x):
        result = _layer_norm_fp32(x, self.weight, self.bias, self.eps)
        return result if torch.is_autocast_enabled() else result.to(x.dtype)


class _SparseNorm(nn.LayerNorm):
    def forward(self, x):
        result = _layer_norm_fp32(x.feats, self.weight, self.bias, self.eps)
        return x.replace(result.to(x.feats.dtype))


def install_tree_norms(model):
    prefixes = ("encoder.nodes.", "decoder.nodes.")
    from ._sparse import volfill_blocks

    encoder, _ = volfill_blocks()
    replaced = 0
    for name, module in list(model.named_modules()):
        if not name.startswith(prefixes) or not isinstance(module, nn.LayerNorm):
            continue
        if tuple(module.normalized_shape) not in ((8,), (16,)):
            continue
        if type(module) is nn.LayerNorm:
            norm = _DenseNorm
        elif type(module) is encoder._SparseLayerNorm32:
            norm = _SparseNorm
        else:
            raise TypeError(f"unsupported normalization at {name}: {type(module).__name__}")
        replacement = norm(module.normalized_shape, eps=module.eps)
        replacement.weight, replacement.bias = module.weight, module.bias
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, replacement)
        replaced += 1
    if replaced != 14:
        raise ValueError(f"expected 14 narrow layer normalizations, found {replaced}")

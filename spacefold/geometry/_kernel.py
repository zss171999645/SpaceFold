"""Tiled FP32 forward kernel for the eight- and sixteen-channel tree features."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n_rows"])
def layer_norm_tile(
    inputs,
    weights,
    biases,
    output,
    n_rows,
    channels: tl.constexpr,
    eps: tl.constexpr,
    tile_rows: tl.constexpr,
):
    row = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    col = tl.arange(0, channels)
    offsets = row[:, None] * channels + col[None, :]
    active = row[:, None] < n_rows
    x = tl.load(inputs + offsets, active, 0)
    mean = tl.sum(x, axis=1) / channels
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / channels
    inverse_std = tl.rsqrt(variance + eps)
    weight = tl.load(weights + col)
    bias = tl.load(biases + col)
    normalized = centered * inverse_std[:, None] * weight[None, :] + bias[None, :]
    tl.store(output + offsets, normalized, active)

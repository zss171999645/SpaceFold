"""VolFill integration and halo-based evaluation of large sparse feature maps."""

from __future__ import annotations

import torch


def volfill_blocks():
    from volfill.amodal.model.vae import latent_vae_sparse_decoder as decoder
    from volfill.amodal.model.vae import latent_vae_sparse_encoder as encoder

    return encoder, decoder


def indexed_residual_block(channels, indice_key):
    """Share one spconv submanifold index map within each tree level."""
    encoder, _ = volfill_blocks()
    block = encoder.SparseConvResBlock(channels)
    for conv in (block.conv1.conv, block.conv2.conv):
        conv.indice_key = indice_key
    return block


def run_spatial_blocks(blocks, sparse, resolution, max_nodes=4_000_000):
    """Evaluate large levels in windows with a two-voxel halo per residual block."""
    if len(sparse.feats) <= max_nodes:
        for block in blocks:
            sparse = block(sparse)
        return sparse

    encoder, _ = volfill_blocks()
    coords, features = sparse.coords, sparse.feats
    count, channels = features.shape
    halo = 2 * len(blocks)
    pieces, output_rows = [], []

    def visit(central_rows, support_rows, lower, upper):
        if not len(central_rows):
            return
        if len(support_rows) > max_nodes:
            lengths = [high - low for low, high in zip(lower, upper)]
            axis = max(range(3), key=lengths.__getitem__)
            if lengths[axis] <= 1:
                raise ValueError("a spatial window and its halo exceed the node budget")
            cut = (lower[axis] + upper[axis]) // 2
            center_axis = coords[central_rows, axis + 1]
            support_axis = coords[support_rows, axis + 1]
            left_center = central_rows[center_axis < cut]
            right_center = central_rows[center_axis >= cut]
            if len(left_center):
                left_upper = list(upper)
                left_upper[axis] = cut
                visit(left_center, support_rows[support_axis < cut + halo], lower, left_upper)
            if len(right_center):
                right_lower = list(lower)
                right_lower[axis] = cut
                visit(right_center, support_rows[support_axis >= cut - halo], right_lower, upper)
            return

        original = coords[support_rows]
        low = torch.tensor(lower, device=coords.device, dtype=coords.dtype)
        high = torch.tensor(upper, device=coords.device, dtype=coords.dtype)
        central = ((original[:, 1:] >= low) & (original[:, 1:] < high)).all(dim=1)
        rows = support_rows[central]
        if len(rows) != len(central_rows):
            raise ValueError("spatial window changed its output membership")
        local = original.clone()
        local[:, 0] = 0
        local[:, 1:] -= (low - halo).clamp_min(0)
        window = encoder.SparseTensor(
            feats=features[support_rows], coords=local, shape=torch.Size((1, channels))
        )
        for block in blocks:
            window = block(window)
        pieces.append(window.feats[central])
        output_rows.append(rows)

    for batch_index in range(int(sparse.shape[0])):
        rows = (coords[:, 0] == batch_index).nonzero(as_tuple=False)[:, 0]
        visit(rows, rows, (0, 0, 0), (resolution,) * 3)
    order = torch.cat(output_rows)
    if len(order) != count:
        raise ValueError("spatial windows did not cover all input rows")
    seen = torch.zeros(count, dtype=torch.bool, device=coords.device)
    seen[order] = True
    if not bool(seen.all()):
        raise ValueError("spatial windows duplicated or omitted an input row")
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(count, device=coords.device)
    return sparse.replace(torch.cat(pieces, dim=0)[inverse])

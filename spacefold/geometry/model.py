"""SpaceFold's R64–R1024 octree VAE with a dense 16 × 32³ bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ._grid import CHILD_OFFSETS_ZYX
from ._sparse import indexed_residual_block, run_spatial_blocks, volfill_blocks
from .octree import UnifiedTUDF

RESOLUTIONS = (64, 128, 256, 512, 1024)
CHANNELS = (128, 64, 32, 16, 8)
LATENT_SHAPE = (16, 32, 32, 32)
MAX_NODES_PER_LEVEL = 127_741_384


class OctreeNodeBudgetExceeded(RuntimeError):
    """A decoder split would exceed the configured per-level node budget."""


@dataclass
class TensorTreeLevel:
    resolution: int
    coords: torch.Tensor
    values: torch.Tensor
    split: torch.Tensor

    def to(self, device):
        return TensorTreeLevel(
            self.resolution, self.coords.to(device), self.values.to(device), self.split.to(device)
        )


@dataclass
class TreeBatch:
    levels: tuple[TensorTreeLevel, ...]
    sample_ids: tuple[str, ...]

    def to(self, device):
        return TreeBatch(tuple(level.to(device) for level in self.levels), self.sample_ids)

    @classmethod
    def from_trees(cls, trees: list[UnifiedTUDF]) -> TreeBatch:
        if not trees:
            raise ValueError("expected at least one octree")
        coords_by_depth = [[] for _ in RESOLUTIONS]
        values_by_depth = [[] for _ in RESOLUTIONS]
        splits_by_depth = [[] for _ in RESOLUTIONS]
        ids = []
        for batch_index, tree in enumerate(trees):
            tree.validate()
            if (
                tuple(level.resolution for level in tree.levels) != RESOLUTIONS
                or tree.truncation_voxels != 3
            ):
                raise ValueError("expected an R64–R1024 octree with truncation at three voxels")
            ids.append(str(tree.provenance.get("sample_id", batch_index)))
            for depth, (level, coords) in enumerate(tree.iter_coords()):
                batch_column = np.full((len(coords), 1), batch_index, dtype=np.int32)
                coords_by_depth[depth].append(
                    torch.from_numpy(np.concatenate((batch_column, coords), axis=1))
                )
                values = level.values.astype(np.float32) * (2 / 3) - 1
                values_by_depth[depth].append(torch.from_numpy(values))
                splits_by_depth[depth].append(torch.from_numpy(level.split))
        levels = tuple(
            TensorTreeLevel(resolution, torch.cat(coords), torch.cat(values), torch.cat(split))
            for resolution, coords, values, split in zip(
                RESOLUTIONS, coords_by_depth, values_by_depth, splits_by_depth
            )
        )
        return cls(levels, tuple(ids))


def _child_coords_bzyx(parents):
    offsets = torch.as_tensor(CHILD_OFFSETS_ZYX, device=parents.device)
    batch = parents[:, None, :1].expand(-1, 8, -1)
    children = parents[:, None, 1:] * 2 + offsets[None]
    return torch.cat((batch, children), dim=2).reshape(-1, 4).int()


class ChildFeatureExpander(nn.Module):
    """Map each selected parent to eight child features in ZYX raster order."""

    def __init__(self, parent_channels, child_channels):
        super().__init__()
        self.child_channels = child_channels
        self.mlp = nn.Sequential(
            nn.LayerNorm(parent_channels),
            nn.Linear(parent_channels, 2 * parent_channels),
            nn.SiLU(),
            nn.Linear(2 * parent_channels, 8 * child_channels),
        )

    def forward(self, parents):
        # Each row is independent. Chunking bounds temporary MLP activations.
        return torch.cat(
            [self.mlp(chunk).reshape(-1, self.child_channels) for chunk in parents.split(65_536)]
        )


class NodeEncoder(nn.Module):
    def __init__(self, channels, parent_channels, resolution, spatial_blocks):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.value_embedding = nn.Sequential(
            nn.Linear(1, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.merge = (
            None
            if parent_channels is None
            else nn.Sequential(
                nn.LayerNorm(8 * channels),
                nn.Linear(8 * channels, parent_channels),
                nn.SiLU(),
                nn.Linear(parent_channels, parent_channels),
            )
        )
        self.context = nn.ModuleList(
            [
                indexed_residual_block(channels, f"unified_encoder_r{resolution}")
                for _ in range(spatial_blocks)
            ]
        )

    def forward(self, level, child_summary, batch_size):
        features = self.value_embedding(level.values[:, None])
        if child_summary is not None:
            selected = level.split.nonzero(as_tuple=False)[:, 0]
            if child_summary.shape != (len(selected), self.channels):
                raise ValueError("child features do not align with split parents")
            features = features.index_add(0, selected, child_summary.to(features.dtype))
        elif bool(level.split.any()):
            raise ValueError("split nodes have no child features")
        if self.context and len(features):
            encoder, _ = volfill_blocks()
            sparse = encoder.SparseTensor(
                feats=features, coords=level.coords, shape=torch.Size((batch_size, self.channels))
            )
            features = run_spatial_blocks(self.context, sparse, self.resolution).feats
        return (
            features if self.merge is None else self.merge(features.reshape(-1, 8 * self.channels))
        )


class UnifiedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder, _ = volfill_blocks()
        self.nodes = nn.ModuleList(
            NodeEncoder(channels, CHANNELS[i - 1] if i else None, resolution, blocks)
            for i, (resolution, channels, blocks) in enumerate(
                zip(RESOLUTIONS, CHANNELS, (0, 2, 2, 1, 0))
            )
        )
        self.core = encoder.SparseTUDFEncoder(
            in_channels=128,
            latent_channels=16,
            channels=[128, 256, 512],
            num_res_blocks=2,
            num_res_blocks_middle=2,
            target_resolution=32,
        )
        if not isinstance(self.core.enc_blocks[6], encoder.SparseDownsample):
            raise TypeError("the installed VolFill encoder has a different downsampling layout")
        self.core.enc_blocks[6] = nn.Identity()

    def forward(self, batch):
        if tuple(level.resolution for level in batch.levels) != RESOLUTIONS:
            raise ValueError("incorrect encoder tree levels")
        # spconv uses the current CUDA context for its index construction.
        with torch.cuda.device_of(batch.levels[0].coords):
            return self._encode(batch)

    def _encode(self, batch):
        summary = None
        for module, level in reversed(list(zip(self.nodes, batch.levels))):
            summary = module(level, summary, len(batch.sample_ids))
        encoder, _ = volfill_blocks()
        root = encoder.SparseTensor(
            feats=summary,
            coords=batch.levels[0].coords,
            shape=torch.Size((len(batch.sample_ids), CHANNELS[0])),
        )
        return self.core(root)


@dataclass
class DecodedTreeLevel:
    resolution: int
    coords: torch.Tensor
    values: torch.Tensor
    split_logits: torch.Tensor | None
    split: torch.Tensor


class NodeDecoder(nn.Module):
    def __init__(self, resolution, channels, child_channels):
        super().__init__()
        self.resolution = resolution
        self.channels = channels
        # The dense trunk supplies the two residual blocks at R64.
        self.context = nn.ModuleList(
            [
                indexed_residual_block(channels, f"unified_decoder_r{resolution}")
                for _ in range(0 if resolution == 64 else 2)
            ]
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1)
        )
        self.split_head = (
            None
            if child_channels is None
            else nn.Sequential(nn.LayerNorm(channels), nn.SiLU(), nn.Linear(channels, 1))
        )
        self.expand = (
            None if child_channels is None else ChildFeatureExpander(channels, child_channels)
        )
        if self.split_head is not None:
            nn.init.constant_(self.split_head[-1].bias, -1)

    def forward(self, features, coords, batch_size):
        if self.context:
            encoder, _ = volfill_blocks()
            sparse = encoder.SparseTensor(
                feats=features, coords=coords, shape=torch.Size((batch_size, self.channels))
            )
            features = run_spatial_blocks(self.context, sparse, self.resolution).feats
        values = self.value_head(features)[:, 0]
        logits = None if self.split_head is None else self.split_head(features)[:, 0]
        return features, values, logits


class SameResolutionProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(512, 256, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class UnifiedDecoder(nn.Module):
    def __init__(self, max_nodes):
        super().__init__()
        _, decoder = volfill_blocks()
        self.max_nodes = int(max_nodes)
        if self.max_nodes < 64**3:
            raise ValueError("node budget must allow the full R64 root")
        self.input_layer = nn.Conv3d(16, 512, 3, padding=1)
        self.middle_block = nn.Sequential(
            *[decoder.ResBlock3d(512, norm_type="layer") for _ in range(2)]
        )
        self.trunk = nn.ModuleList()
        for channels in (512, 256, 128):
            self.trunk.extend(decoder.ResBlock3d(channels, norm_type="layer") for _ in range(2))
            if channels == 512:
                self.trunk.append(SameResolutionProjection())
            elif channels == 256:
                self.trunk.append(decoder.UpsampleBlock3d(256, 128))
        self.nodes = nn.ModuleList(
            NodeDecoder(resolution, channels, CHANNELS[i + 1] if i + 1 < len(CHANNELS) else None)
            for i, (resolution, channels) in enumerate(zip(RESOLUTIONS, CHANNELS))
        )

    def forward(self, z, *, teacher: TreeBatch | None = None):
        if z.ndim != 5 or tuple(z.shape[1:]) != LATENT_SHAPE:
            raise ValueError("expected a latent of shape [B, 16, 32, 32, 32]")
        with torch.cuda.device_of(z):
            return self._decode(z, teacher)

    def _decode(self, z, teacher):
        h = self.middle_block(self.input_layer(z))
        for block in self.trunk:
            h = block(h)
        batch_size = z.shape[0]
        features = h.permute(0, 2, 3, 4, 1).reshape(-1, CHANNELS[0])
        if teacher is not None:
            if batch_size != len(teacher.sample_ids):
                raise ValueError("teacher batch size mismatch")
            coords = teacher.levels[0].coords
        else:
            axis = torch.arange(64, dtype=torch.int32, device=z.device)
            grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            coords = torch.cat(
                [
                    torch.cat((torch.full_like(grid[:, :1], b), grid), dim=1)
                    for b in range(batch_size)
                ]
            )
        outputs = []
        for index, stage in enumerate(self.nodes):
            features, values, logits = stage(features, coords, batch_size)
            if logits is None:
                split = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
            elif teacher is None:
                split = logits > 0
            else:
                target = teacher.levels[index]
                if target.resolution != stage.resolution or not torch.equal(coords, target.coords):
                    raise ValueError("teacher tree and decoder coordinates differ")
                split = target.split
            outputs.append(DecodedTreeLevel(stage.resolution, coords, values, logits, split))
            if logits is None or not bool(split.any()):
                break
            child_count = int(split.sum()) * 8
            if child_count > self.max_nodes:
                raise OctreeNodeBudgetExceeded(
                    f"R{stage.resolution * 2} requires {child_count} nodes, exceeding the limit of {self.max_nodes}"
                )
            features = stage.expand(features[split])
            coords = _child_coords_bzyx(coords[split])
        return tuple(outputs)


class UnifiedOctreeVAE(nn.Module):
    """Encode a batch of five-level octrees and decode their native surface fields."""

    def __init__(self, *, max_nodes=MAX_NODES_PER_LEVEL):
        super().__init__()
        self.encoder = UnifiedEncoder()
        self.decoder = UnifiedDecoder(max_nodes=max_nodes)

    def forward(self, batch: TreeBatch, *, use_gt_topology=True, sample_posterior=False):
        mean, logvar = self.encoder(batch)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample_posterior else mean
        return self.decoder(z, teacher=batch if use_gt_topology else None), mean, logvar

# Geometry codec

The geometry encoder reads a five-level octree containing truncated unsigned distances at resolutions 64, 128, 256, 512 and 1024. It combines the ordered features of each group of eight children and maps the root to a dense posterior with 16 channels on a 32 × 32 × 32 grid. The decoder predicts a distance and a split decision at each eligible node. Surface extraction returns the centers of terminal voxels in world XYZ coordinates.

## Build an input tree

For a scene represented by surface points, create a NumPy NPZ file with these arrays.

| Key | Shape | Meaning |
| --- | --- | --- |
| `points` | `(N, 3)` | Finite world XYZ coordinates |
| `bbox_min` | `(3,)` | Lower XYZ corner, in the same units |
| `bbox_max` | `(3,)` | Upper XYZ corner, in the same units |

Choose a cube around the scene for isotropic voxels. The builder uses every input voxel inside the half-open bounds and measures distance in each level’s voxel units, with truncation at three voxels. It checks that the final tree retains every occupied input voxel. The CPU builder materializes a 1024³ grid and requires several GiB of host memory.

```bash
python -m spacefold.geometry build --input points.npz --output scene.tree.npz
```

For an existing three-level envelope in the `octree_bitmask_palette4_v2` format, the data-preparation entry point reads the R256 base distance field, its refinement masks, and the R512/R1024 child distances. It adds the two coarser levels and applies the same surface-preserving pruning used to prepare the codec’s training trees.

```bash
python -m spacefold.geometry prepare --input envelope.npz --output scene.tree.npz
```

Both commands write a new tree file. Use a distinct output name for each input.

## Encode and decode

Use an NVIDIA CUDA device with the VolFill sparse modules, spconv and Triton installed. Pass the VolFill checkout with `--volfill-root`. This option is optional when its Python modules are already importable. The CLI takes the SpaceFold geometry safetensors checkpoint. Run these commands from an environment where the `spacefold` package is importable.

```bash
python -m spacefold.geometry encode \
  --weights weights/spacefold-geometry.safetensors --volfill-root external/VolFill \
  --input scene.tree.npz --output scene.latent.npz

python -m spacefold.geometry decode \
  --weights weights/spacefold-geometry.safetensors --volfill-root external/VolFill \
  --input scene.latent.npz --output reconstructed.points.npz
```

The latent archive contains the FP16 posterior mean under `raw`, with shape `(16, 32, 32, 32)`, and the two XYZ bounding-box corners. Decoding uses the model’s predicted topology. Its output contains `points`, `bbox_min` and `bbox_max`. Each row of `points` is a float32 voxel center in the original frame. A panorama-generated latent archive with the same `raw` and bounding-box fields can also be decoded.

For direct use in Python, `TreeBatch.from_trees([tree])` constructs the model input. Each row of its coordinate tensor is ordered as batch, Z, Y, X. The encoder’s values are normalized with `2 * distance / 3 - 1`. A terminal decoder value at or below `-2/3` corresponds to a surface threshold of half a voxel.

```python
import torch

from spacefold.geometry import TreeBatch, UnifiedTUDF, load_geometry_checkpoint, register_volfill_source
from spacefold.geometry.surface import predicted_surface_points

register_volfill_source("external/VolFill")
tree = UnifiedTUDF.load("scene.tree.npz")
model, metadata = load_geometry_checkpoint("weights/spacefold-geometry.safetensors", "cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    posterior_mean, posterior_logvar = model.encoder(TreeBatch.from_trees([tree]).to("cuda"))
    levels = model.decoder(posterior_mean, teacher=None)
points, counts = predicted_surface_points(levels, bbox_min=tree.bbox_min, bbox_max=tree.bbox_max)
```

SpaceFold uses VolFill’s sparse TUDF encoder and dense decoder blocks through its Python modules. The sparse residual stages reuse VolFill’s submanifold convolutions with one spconv index map per resolution. See the pinned [VolFill source](https://github.com/ngoductuanlhp/VolFill/tree/a0c8e7779ccee318e7a75bb2d273290d4ffb76ed).

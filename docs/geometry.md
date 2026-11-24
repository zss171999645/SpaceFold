# Geometry codec

The geometry encoder reads a five-level octree containing truncated unsigned distances at resolutions 64, 128, 256, 512 and 1024. It combines the ordered features of each group of eight children and maps the root to a dense posterior with 16 channels on a 32 × 32 × 32 grid. The decoder predicts a distance and a split decision at each eligible node. Surface extraction returns the centers of terminal voxels in world XYZ coordinates.

## Build an input tree

For a scene represented by surface points, create a NumPy NPZ file with these arrays.

| Key | Shape | Meaning |
| --- | --- | --- |
| `points` | `(N, 3)` | Finite world XYZ coordinates |
| `bbox_min` | `(3,)` | Lower XYZ corner, in the same units |
| `bbox_max` | `(3,)` | Upper XYZ corner, in the same units |

Choose a cube around the scene for isotropic voxels. The builder keeps points satisfying `bbox_min <= point < bbox_max` on all three axes. Points in the same R1024 voxel share one surface seed. Float64 bounds retain their precision during voxelization. Distances are measured in each level’s voxel units and truncated at three voxels. The builder checks that the final tree retains every occupied input voxel. It materializes a 1024³ grid and requires several GiB of host memory.

```bash
python -m spacefold.geometry build --input points.npz --output scene.tree.npz
```

For an existing three-level envelope in the `octree_bitmask_palette4_v2` format, the data-preparation entry point reads the R256 base distance field, its refinement masks, and the R512/R1024 child distances. It adds the two coarser levels and applies the same surface-preserving pruning used to prepare the codec’s training trees.

```bash
python -m spacefold.geometry prepare --input envelope.npz --output scene.tree.npz
```

The compact envelope uses the same XYZ bounds as a point archive. Its scalar fields are `storage_format="octree_bitmask_palette4_v2"`, `base_resolution=256` and `truncation_voxels=3`. The base grid and the eight children within each refined node use ZYX raster order with X varying fastest.

| Array | Contents |
| --- | --- |
| `base_tudf` | Float16 distances on the full `(256, 256, 256)` grid, from zero to three voxels. |
| `refine_base_mask_bits` | A flat uint8 array of length `256³ / 8`. The low bit marks the first base voxel in each group of eight. |
| `refine_middle_child_mask` | One uint8 per refined R256 parent, in base raster order. Bit `4*dz + 2*dy + dx` marks the R512 child to split. |
| `middle_children_codes4`, `fine_children_codes4` | Four uint8 values per refined parent at R256 and R512 respectively. Each byte encodes two child distances, low nibble first. R512 parents follow the order of the set bits in `refine_middle_child_mask`, scanning each byte from low to high. |
| `tudf_palette` | The float16 codebook in ascending code order is `[0, 1, √2, √3, 2, √5, √6, √8, 3]`. |

Both commands write a new tree file. Use a distinct output name for each input.

## Encode and decode

Use an NVIDIA CUDA device with the VolFill sparse modules, spconv and Triton installed. Run `spacefold-setup --geometry-only` once to prepare the pinned VolFill source. The CLI then locates it automatically. To select another existing checkout, pass `--volfill-root`. An already importable VolFill installation also works when the managed checkout is absent. The CLI takes the SpaceFold geometry safetensors checkpoint. Run these commands from an environment where the `spacefold` package is importable.

```bash
python -m spacefold.geometry encode \
  --weights weights/spacefold-geometry.safetensors \
  --input scene.tree.npz --output scene.latent.npz

python -m spacefold.geometry decode \
  --weights weights/spacefold-geometry.safetensors \
  --input scene.latent.npz --output reconstructed.points.npz
```

Use `--device cuda:1` on either command to select the second visible CUDA device. In Python, put the model, its input tree batch and its input latent on the intended device.

The latent archive contains the FP16 posterior mean under `raw`, with shape `(16, 32, 32, 32)`, and the two XYZ bounding-box corners. Repeated CUDA encodes of the same tree can yield slightly different FP16 values. Retain the raw latent to decode that particular sample again.

Decoding uses the model’s predicted topology. Its output contains `points`, `bbox_min` and `bbox_max`. Each row of `points` is a float32 voxel center in the original frame. A panorama-generated latent archive with the same `raw` and bounding-box fields can also be decoded.

For direct use in Python, `TreeBatch.from_trees([tree])` constructs the model input. Each row of its coordinate tensor is ordered as batch, Z, Y, X. The encoder’s values are normalized with `2 * distance / 3 - 1`. A terminal decoder value at or below `-2/3` corresponds to a surface threshold of half a voxel.

```python
import torch

from spacefold.dependencies import VOLFILL, cached_source
from spacefold.geometry import TreeBatch, UnifiedTUDF, load_geometry_checkpoint, register_volfill_source
from spacefold.geometry.surface import predicted_surface_points

register_volfill_source(cached_source(VOLFILL))
tree = UnifiedTUDF.load("scene.tree.npz")
model, metadata = load_geometry_checkpoint("weights/spacefold-geometry.safetensors", "cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    posterior_mean, posterior_logvar = model.encoder(TreeBatch.from_trees([tree]).to("cuda"))
    levels = model.decoder(posterior_mean, teacher=None)
points, counts = predicted_surface_points(levels, bbox_min=tree.bbox_min, bbox_max=tree.bbox_max)
```

SpaceFold uses VolFill’s sparse TUDF encoder and dense decoder blocks through its Python modules. The sparse residual stages reuse VolFill’s submanifold convolutions with one spconv index map per resolution. See the pinned [VolFill source](https://github.com/ngoductuanlhp/VolFill/tree/a0c8e7779ccee318e7a75bb2d273290d4ffb76ed).

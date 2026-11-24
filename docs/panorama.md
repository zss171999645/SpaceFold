# Panorama-conditioned scene generation

SpaceFold takes one **level, 2:1 equirectangular RGB image** and a camera height above a horizontal floor. Use a measured height when available. An assumed height determines the scale of the generated room. Output coordinates are in metres, with the panorama centre looking along +y, image-right along +x, and +z up. The camera is at `(0, 0, camera_height)`.

## Inputs

| Input | Expected contents |
| --- | --- |
| Panorama | A level indoor panorama in a Pillow-supported image format. The pipeline applies EXIF orientation, checks the displayed 2:1 aspect ratio and resizes the RGB image to 2048 × 1024. |
| Generator | SpaceFold safetensors containing the trained DiT and the panorama token projector. |
| Geometry VAE | SpaceFold geometry safetensors for the 16-channel, 32³ scene latent. |
| Latent statistics | Numeric NPZ containing `mean` and `std`, each float32 with shape `(16,)`. Use the statistics accompanying the checkpoints. |
| DA² | The DA-2 source checkout and the official `model.safetensors`. |
| VolFill | The VolFill source checkout supplying the 3D transformer architecture. The trained transformer parameters come from the SpaceFold generator file. |

The reference DA² source revision is `d659838585f2bc9967c7e9367af573795271dbbb`. The separately obtained model has SHA-256 `d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e`. [DA² source](https://github.com/EnVision-Research/DA-2/tree/d659838585f2bc9967c7e9367af573795271dbbb) · [Official model revision](https://huggingface.co/haodongli/DA-2/tree/0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3)

The reference VolFill source revision is `a0c8e7779ccee318e7a75bb2d273290d4ffb76ed`. In the reference environment, DA² and VolFill's dense 3D DiT use PyTorch scaled dot-product attention. If xFormers is already installed, DA² can select its CUDA attention implementation. Consult each upstream project's terms for the components you obtain. [VolFill source](https://github.com/ngoductuanlhp/VolFill/tree/a0c8e7779ccee318e7a75bb2d273290d4ffb76ed) · [DA² attention selection](https://github.com/EnVision-Research/DA-2/blob/d659838585f2bc9967c7e9367af573795271dbbb/src/da2/model/dinov2/attention.py)

Run `spacefold-setup --with-da2-weights` once to prepare the official sources and DA² checkpoint. Explicit paths supplied through `--volfill-root`, `--da2-root` or `--da2-checkpoint` take precedence.

Run on a CUDA GPU with native BF16 support. The geometry encoder and decoder run in FP16, and visible pixel voxelization and Euclidean distance transforms run in host memory. Choose a fresh output directory.

```bash
python -m spacefold.panorama.infer \
  --panorama room.jpg \
  --camera-height 1.6 \
  --dit-checkpoint weights/spacefold-panorama.safetensors \
  --vae-checkpoint weights/spacefold-geometry.safetensors \
  --latent-stats weights/latent_stats.npz \
  --steps 50 --guidance 3 --seed 0 \
  --output outputs/room
```

## Reading the result

Open `generated_geometry.ply` in a point-cloud viewer. Each point is the centre of a terminal octree cell predicted to contain surface geometry. The cloud uses a uniform grey display colour. To process the same geometry in NumPy, load `prediction_points.npz`. Its `points` field is a float32 `(N, 3)` array of xyz positions, with `bbox_min` and `bbox_max` in the same frame.

SpaceFold stages the output beside its destination and publishes it when all files are ready. Filesystems that support an exclusive directory rename expose the entire result in one operation. On a filesystem that requires a separate reservation, terminating the process during the final rename can leave an empty destination. Choose a fresh output path for that retry.

The intermediate files document each stage of reconstruction.

| File | Contents |
| --- | --- |
| `scene.json` | SHA-256 of the original panorama file, camera pose and height, fitted relative-to-metric distance scale, bounds, sampler settings and point counts by octree resolution. |
| `da2_distance_fp32.npy` | Relative radial distance at the native 546 × 1092 pixel centres. |
| `visible_native_tree.npz` | The octree built from all in-bounds visible depth pixels. |
| `visible_latent_fp16.npy` | Posterior mean of the visible octree, rounded to FP16 as during training. |
| `generated_latent.npz` | Normalized and raw float32 `(16, 32, 32, 32)` latents, plus `bbox_min` and `bbox_max`. Use `raw` with the geometry decoder. |

## Sampling from existing conditioning tensors

The image pipeline takes the 90th percentile of the vertical drop along downward rays whose vertical component has magnitude at least 0.25. Dividing the supplied camera height by this drop converts relative radial distance to metres. SpaceFold fits a bounding cube to the resulting points, builds a 64→1024 visible distance octree and encodes it with the geometry VAE.

For the image condition, SpaceFold transforms 2,048 fixed Fibonacci-sphere directions into the camera frame and bilinearly samples three DA² feature maps with horizontal wraparound. The 512, 256 and 128 channels form an 896-dimensional vector. A learned LayerNorm and linear projection produce 768 channels. For a unit target-frame direction `(x, y, z)`, the trained directional encoding uses latitude `asin(y)` and longitude `atan2(x, z)`. Its four 192-channel blocks encode sine and cosine of latitude, then sine and cosine of longitude, at integer frequencies 1 through 192. SpaceFold adds this encoding to the projected feature. The single-image command treats every input pixel as observed. The resulting tokens feed cross-attention in the 960-channel DiT. The sampler also receives the normalized visible latent at every Euler step.

Applications that already have compatible tensors can call the sampler directly. `image_tokens` must contain the trained projector's **BF16** outputs in target-frame ray order, shape `(1, 2048, 768)`. `visible_mean` must contain the geometry encoder's **unnormalized** posterior mean, shape `(16, 32, 32, 32)`, in the same scene frame. This path loads the SpaceFold DiT weights and VolFill architecture and reads no panorama or DA² checkpoint.

```python
from pathlib import Path

import torch

from spacefold.dependencies import VOLFILL, cached_source
from spacefold.panorama.model import load_generator, load_latent_stats, sample_latent

device = torch.device("cuda:0")
generator = load_generator(
    Path("weights/spacefold-panorama.safetensors"), cached_source(VOLFILL), device
)
mean, std = load_latent_stats(Path("weights/latent_stats.npz"))
visible = (visible_mean.detach().cpu().half().float() - mean) / std
normalized = sample_latent(
    generator,
    visible[None].to(device),
    image_tokens.to(device),
    steps=50,
    guidance=3.0,
    seed=0,
)
raw = normalized * std + mean
```

The sampler returns a CPU float32 tensor. Use the bounds associated with the input visible scene when decoding its raw latent. The seed fixes the starting CUDA noise. GPU kernels and library versions can affect the decoded surface even when the seed and input files are identical.

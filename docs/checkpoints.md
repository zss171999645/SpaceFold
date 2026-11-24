# Model files

The single-image pipeline takes two SpaceFold state files, the normalization statistics paired with the generator, and the external DA² model. We will release the two SpaceFold checkpoints and their matching statistics in a future update. Their inference formats are listed below.

| Input | Contents |
|---|---|
| `spacefold-geometry.safetensors` | Complete geometry encoder and decoder state. The safetensors metadata field `format` is `spacefold.geometry.v1`. Tensor names match the geometry codec's state dictionary. |
| `spacefold-panorama.safetensors` | The latent transformer and trained image-token projector. The metadata field `format` is `spacefold.panorama.v1`. The `dit.` and `conditioner_adapter.` prefixes identify the two components. The reference inference uses the EMA branch. |
| `latent_stats.npz` | Two ordinary NumPy arrays, `mean` and `std`. Each has dtype float32, shape `(16,)`, and finite entries. Every standard deviation exceeds 1e-6. Compute normalized latents by subtracting the channel mean and dividing by the channel standard deviation. Both visible and target latents use the generator's train-split statistics. |
| `da2.safetensors` | The official DA² parameter file for the frozen panorama network. Obtain it directly under the publisher's terms. [Model card](https://huggingface.co/haodongli/DA-2/blob/0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3/README.md) |

For the external DA² model, the tested revision is `0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3`. Its `model.safetensors` file has 1,378,513,064 bytes and SHA-256 `d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e`. To download the public DA² file and its pinned source dependencies, use the explicit setup command.

```bash
spacefold-setup --with-da2-weights
spacefold-setup --check --with-da2-weights
```

The second command verifies the chosen cache offline. Inference reads the cached DA² file automatically. If you already obtained that file, pass its path with `--da2-checkpoint`. Use `--da2-root` and `--volfill-root` to select existing source directories.

The SpaceFold codec produces a 16-channel latent on a 32³ grid. The reference generator has 16 transformer blocks, width 960 and 15 attention heads. Its 32 input channels concatenate a noisy target latent and a normalized visible latent. SpaceFold imports model layers from the VolFill source checkout. VolFill's published weights and 16³ training statistics belong to its own model.

See [third-party terms](../THIRD_PARTY.md) and [the panorama interface](panorama.md).

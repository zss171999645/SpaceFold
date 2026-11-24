# Model files

The single-image pipeline takes two SpaceFold state files, the normalization statistics paired with the generator, and the external DA² model. SpaceFold's learned assets are subject to a separate release decision. Use an authorized copy of the three matched SpaceFold files. The following interface describes the tensor-only inference format.

| Input | Contents |
|---|---|
| `spacefold-geometry.safetensors` | Complete geometry encoder and decoder state. The safetensors metadata field `format` is `spacefold.geometry.v1`. Tensor names match the geometry codec's state dictionary. |
| `spacefold-panorama.safetensors` | The latent transformer and trained image-token projector. The metadata field `format` is `spacefold.panorama.v1`. The `dit.` and `conditioner_adapter.` prefixes identify the two components. The reference inference uses the EMA branch. |
| `latent_stats.npz` | Two ordinary NumPy arrays, `mean` and `std`. Each has dtype float32, shape `(16,)`, and finite entries. Every standard deviation exceeds 1e-6. Compute normalized latents by subtracting the channel mean and dividing by the channel standard deviation. Both visible and target latents use the generator's train-split statistics. |
| `da2.safetensors` | The official DA² parameter file for the frozen panorama network. Obtain it directly under the publisher's terms. [Model card](https://huggingface.co/haodongli/DA-2/blob/0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3/README.md) |

For the external DA² model, the tested revision is `0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3`. Its `model.safetensors` file has 1,378,513,064 bytes and SHA-256 `d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e`. The following commands fetch that file from its publisher and check its identity.

```bash
mkdir -p weights
curl --fail --location --output weights/da2.safetensors \
  https://huggingface.co/haodongli/DA-2/resolve/0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3/model.safetensors
printf '%s  %s\n' d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e \
  weights/da2.safetensors | sha256sum --check
```

The SpaceFold codec produces a 16-channel latent on a 32³ grid. The reference generator has 16 transformer blocks, width 960 and 15 attention heads. Its 32 input channels concatenate a noisy target latent and a normalized visible latent. SpaceFold imports model layers from the VolFill source checkout. VolFill's published weights and 16³ training statistics belong to its own model.

See [third-party terms](../THIRD_PARTY.md) and [the panorama interface](panorama.md).

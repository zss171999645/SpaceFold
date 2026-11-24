# Third-party components

SpaceFold calls external model implementations through their Python interfaces. The opt-in `spacefold-setup` command fetches the pinned revisions directly from the publishers into a per-user cache outside this source tree. The optional `--with-da2-weights` flag includes the official DA² checkpoint. Each source and model remains subject to its applicable upstream terms.

| Component | Use in this implementation | Source and terms |
|---|---|---|
| VolFill | Sparse geometry layers and the latent transformer backbone | [Tested source](https://github.com/ngoductuanlhp/VolFill/tree/a0c8e7779ccee318e7a75bb2d273290d4ffb76ed) · [Licence and component notices](https://github.com/ngoductuanlhp/VolFill/blob/a0c8e7779ccee318e7a75bb2d273290d4ffb76ed/LICENSE) |
| DA² | Frozen panorama distance and image features | [Tested source](https://github.com/EnVision-Research/DA-2/tree/d659838585f2bc9967c7e9367af573795271dbbb) · [Source licence](https://github.com/EnVision-Research/DA-2/blob/d659838585f2bc9967c7e9367af573795271dbbb/LICENSE) · [Model card at the tested revision](https://huggingface.co/haodongli/DA-2/tree/0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3) |
| Empty Warehouse 01 | Optional real RGB panorama fetched from the publisher | Sergej Majboroda for Poly Haven. [Asset](https://polyhaven.com/a/empty_warehouse_01) · [Asset licence](https://polyhaven.com/license) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

VolFill identifies MIT-licensed code and lists notices for MoGe, TRELLIS and DINOv2. DA² labels its code and model card Apache-2.0. Parts of its decoder overlap with code in UniK3D, which publishes CC BY-NC-SA 4.0 terms. The downstream permissions for those portions warrant confirmation from the corresponding rights holders, particularly for commercial use or redistribution. The external packages and their learned weights remain separate from this source distribution. [DA² decoder](https://github.com/EnVision-Research/DA-2/blob/d659838585f2bc9967c7e9367af573795271dbbb/src/da2/model/base.py) · [UniK3D layers](https://github.com/lpiccinelli-eth/UniK3D/blob/29f7862f3c3ee79c53ef0153dfc901ce511df575/unik3d/layers/upsample.py) · [UniK3D licence](https://github.com/lpiccinelli-eth/UniK3D/blob/29f7862f3c3ee79c53ef0153dfc901ce511df575/LICENSE).

Python and CUDA dependencies are installed through their own distributions. The licence on a dependency applies to that component. SpaceFold source and SpaceFold-trained weights require a release decision by their rights holders.

"""Load the tensor-only SpaceFold geometry checkpoint."""

from pathlib import Path

from safetensors import safe_open

from ._normalization import install_tree_norms
from .model import LATENT_SHAPE, MAX_NODES_PER_LEVEL, UnifiedOctreeVAE

FORMAT = "spacefold.geometry.v1"


def load_geometry_checkpoint(path: str | Path, device):
    """Return the frozen octree codec and its public architecture metadata."""
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(f"expected a safetensors checkpoint with format={FORMAT!r}")
        names = checkpoint.keys()
        state = {name: checkpoint.get_tensor(name) for name in names}

    model = UnifiedOctreeVAE()
    install_tree_norms(model)
    for name, tensor in model.state_dict().items():
        if name in state and state[name].dtype != tensor.dtype:
            raise ValueError(f"{name} has dtype {state[name].dtype}, expected {tensor.dtype}")
    model.load_state_dict(state, strict=True)
    # spconv uses the training flag to select its numerical execution path.
    # The reference inference uses frozen parameters with this flag set.
    model = model.to(device).train().requires_grad_(False)
    return model, {
        "format": FORMAT,
        "latent_shape": LATENT_SHAPE,
        "max_nodes": MAX_NODES_PER_LEVEL,
    }

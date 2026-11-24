"""Bind the user-supplied VolFill checkout for its absolute Python imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def register_volfill_source(root: str | Path) -> Path:
    """Make one VolFill checkout, including its TRELLIS sparse backend, importable."""
    checkout = Path(root).expanduser().resolve(strict=True)
    package = checkout / "volfill"
    sparse = checkout / "third_party" / "trellis" / "modules" / "sparse"
    for path in (
        package / "__init__.py",
        package / "amodal/model/vae/latent_vae_sparse_encoder.py",
        package / "amodal/model/vae/latent_vae_sparse_decoder.py",
        sparse / "__init__.py",
        sparse / "conv/conv_spconv.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete VolFill checkout, missing {path}")

    for name, module in tuple(sys.modules.items()):
        if name == "volfill" or name.startswith("volfill."):
            expected = package
        elif name == "third_party.trellis.modules.sparse" or name.startswith(
            "third_party.trellis.modules.sparse."
        ):
            expected = sparse
        else:
            continue
        source = getattr(module, "__file__", None)
        if source is None or not Path(source).resolve().is_relative_to(expected):
            raise ImportError(
                f"{name} is already loaded from {source}, expected a module under {expected}"
            )

    entry = str(checkout)
    if not sys.path or sys.path[0] != entry:
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
        importlib.invalidate_caches()
    bound = importlib.import_module("volfill")
    if Path(bound.__file__).resolve() != package / "__init__.py":
        raise ImportError(f"could not bind VolFill from {package}")
    return checkout

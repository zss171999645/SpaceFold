"""Publish complete geometry NPZ files while preserving existing output files."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def save_npz(path, **arrays):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as temporary:
        name = temporary.name
        try:
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.link(name, destination)
        finally:
            os.unlink(name)

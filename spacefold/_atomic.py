"""Publish a completed directory without overwriting existing content."""

import os
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def result_directory(destination: Path):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spacefold-", dir=destination.parent) as staging:
        result = Path(staging) / "result"
        result.mkdir()
        yield result

        # mkdir is exclusive even if another run finished during sampling.
        destination.mkdir()
        reservation = destination.lstat()
        try:
            os.replace(result, destination)
        except BaseException:
            # Preserve the original error and any output written by another process.
            with suppress(OSError):
                if os.path.samestat(reservation, destination.lstat()):
                    destination.rmdir()
            raise

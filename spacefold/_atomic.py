"""Publish a completed directory without overwriting existing content."""

import ctypes
import errno
import os
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory


def _rename_exclusive(source: Path, destination: Path) -> bool:
    """Publish with one no-replace rename when the platform and filesystem support it."""
    if sys.platform == "darwin":
        name, cwd, flag = "renameatx_np", -2, 4  # AT_FDCWD, RENAME_EXCL
    elif sys.platform == "linux":
        name, cwd, flag = "renameat2", -100, 1  # AT_FDCWD, RENAME_NOREPLACE
    else:
        return False
    rename = getattr(ctypes.CDLL(None, use_errno=True), name, None)
    if rename is None:
        return False
    # Preserve the Python audit event that os.rename would issue.
    sys.audit("os.rename", os.fspath(source), os.fspath(destination), -1, -1)
    if rename(cwd, os.fsencode(source), cwd, os.fsencode(destination), flag) == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.ENOSYS, errno.ENOTSUP, errno.EINVAL):
        return False
    raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def result_directory(destination: Path):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spacefold-", dir=destination.parent) as staging:
        result = Path(staging) / "result"
        result.mkdir()
        yield result
        if _rename_exclusive(result, destination):
            return

        # Older filesystems still need an exclusive reservation to preserve existing output.
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

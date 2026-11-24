"""Prepare the pinned public dependencies in a user-owned cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.request import getproxies

from ._atomic import result_directory


@dataclass(frozen=True)
class Source:
    name: str
    repository: str
    revision: str
    files: int
    unpacked_bytes: int
    contents_sha256: str

    def location(self) -> Path:
        return cache_dir() / "sources" / self.name / self.revision


# The fingerprint covers every tracked file's relative name and SHA-256,
# including upstream licences. It is independent of tar ordering and compression.
VOLFILL = Source(
    "VolFill",
    "ngoductuanlhp/VolFill",
    "a0c8e7779ccee318e7a75bb2d273290d4ffb76ed",
    91,
    2_204_642,
    "5afbc007f754fa7db09512f027e9366e64eba52fc7e7488ebccf4a390b46520b",
)
DA2 = Source(
    "DA-2",
    "EnVision-Research/DA-2",
    "d659838585f2bc9967c7e9367af573795271dbbb",
    81,
    17_008_945,
    "9c49d0b6f77dd7256cb6f89a15ad13cae7fff8d9bd1b6740bb63accf0a86906f",
)
MODEL_REPOSITORY = "haodongli/DA-2"
MODEL_REVISION = "0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3"
MODEL_SIZE = 1_378_513_064
MODEL_SHA256 = "d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e"


class DependencyError(RuntimeError):
    """An upstream asset is missing or differs from the tested version."""


def cache_dir() -> Path:
    override = os.environ.get("SPACEFOLD_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and Path(xdg).is_absolute():
        base = Path(xdg)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path.home() / ".cache"
    return (base / "spacefold").resolve()


def _model_path() -> Path:
    return cache_dir() / "weights" / "DA-2" / MODEL_REVISION / "model.safetensors"


def _digest(path: Path) -> bytes:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.digest()


def _verify_source(source: Source, path: Path) -> Path:
    error = DependencyError(
        f"Unexpected contents for {source.name} {source.revision} at {path}. "
        "Keep your copy and choose a fresh SPACEFOLD_CACHE_DIR to prepare the pinned source."
    )
    if path.is_symlink() or not path.is_dir():
        raise error
    entries = {}
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise error
        if item.is_dir() or item.name == ".DS_Store":
            continue
        if item.parent.name == "__pycache__" and item.suffix == ".pyc":
            continue
        if not item.is_file():
            raise error
        total += item.stat().st_size
        if total > source.unpacked_bytes or len(entries) >= source.files:
            raise error
        entries[item.relative_to(path).as_posix()] = _digest(item)
    fingerprint = hashlib.sha256()
    for relative, digest in sorted(entries.items()):
        fingerprint.update(relative.encode("utf-8") + b"\0" + digest)
    if (len(entries), total, fingerprint.hexdigest()) != (
        source.files,
        source.unpacked_bytes,
        source.contents_sha256,
    ):
        raise error
    return path


def cached_source(source: Source) -> Path:
    """Locate and verify an official source snapshot without contacting the network."""
    path = source.location()
    if not os.path.lexists(path):
        command = "spacefold-setup --geometry-only" if source == VOLFILL else "spacefold-setup"
        raise DependencyError(f"Missing {source.name} at {path}. Run {command}.")
    return _verify_source(source, path)


def _fetch_source(source: Source) -> None:
    url = f"https://codeload.github.com/{source.repository}/tar.gz/{source.revision}"
    prefix = f"{source.name}-{source.revision}"
    print(f"Fetching {source.name} {source.revision} from {url}", flush=True)
    print(
        f"Upstream terms: https://github.com/{source.repository}/blob/{source.revision}/LICENSE",
        flush=True,
    )
    _outside_project(source.location())
    with result_directory(source.location()) as stage:
        archive = stage.parent / "source.tar.gz"
        _transfer_or_raise(url, archive, source.unpacked_bytes + 2_000_000)
        total = 0
        count = 0
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                name = PurePosixPath(member.name)
                if (
                    not name.parts
                    or name.parts[0] != prefix
                    or ".." in name.parts
                    or "\\" in member.name
                ):
                    raise DependencyError(f"Unexpected archive path: {member.name}")
                if member.isdir():
                    continue
                total += member.size
                count += 1
                if not member.isreg() or total > source.unpacked_bytes or count > source.files:
                    raise DependencyError(f"Unexpected archive entry: {member.name}")
                target = stage.joinpath(*name.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as inp, target.open("xb") as out:
                    shutil.copyfileobj(inp, out)
                if member.mode & 0o111:
                    target.chmod(0o755)
        _verify_source(source, stage)


def _verify_weight(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        raise DependencyError(f"Expected {MODEL_SIZE:,} bytes of DA² weights at {path}")
    if _digest(path).hex() != MODEL_SHA256:
        raise DependencyError(f"DA² weight SHA-256 mismatch at {path}")
    return path


def cached_da2_checkpoint() -> Path:
    """Locate and verify the publisher's checkpoint without downloading it."""
    path = _model_path()
    if not os.path.lexists(path):
        raise DependencyError(
            f"Missing DA² weights at {path}. Run spacefold-setup --with-da2-weights."
        )
    return _verify_weight(path)


def panorama_paths(
    *,
    volfill_root: Path | None = None,
    da2_root: Path | None = None,
    da2_checkpoint: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the two source roots and the DA² model offline, preserving explicit overrides."""
    volfill = volfill_root if volfill_root is not None else cached_source(VOLFILL)
    depth = da2_root if da2_root is not None else cached_source(DA2)
    checkpoint = da2_checkpoint if da2_checkpoint is not None else cached_da2_checkpoint()
    for root in (volfill, depth):
        if not root.is_dir():
            raise DependencyError(f"Missing upstream source directory: {root}")
    if not checkpoint.is_file():
        raise DependencyError(f"Missing DA² checkpoint: {checkpoint}")
    return volfill, depth, checkpoint


def _transfer(url: str, destination: Path, limit: int, *, resume: bool = False) -> tuple[int, str]:
    args = [
        "curl",
        "--disable",
        "--fail",
        "--location",
        "--show-error",
        "--retry",
        "2",
        "--connect-timeout",
        "15",
        "--speed-limit",
        "1024",
        "--speed-time",
        "90",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--max-filesize",
        str(limit),
        "--progress-bar" if sys.stderr.isatty() else "--silent",
        "--output",
        str(destination),
        "--write-out",
        "%{response_code}",
    ]
    if resume:
        args.extend(["--continue-at", "-"])
    # curl reads proxy variables on Unix. Python also knows macOS system proxy settings.
    env = os.environ.copy()
    for scheme, address in getproxies().items():
        name = f"{scheme}_proxy"
        if scheme in ("https", "no") and name not in env and name.upper() not in env:
            env[name] = address
    try:
        result = subprocess.run(
            [*args, url], stdout=subprocess.PIPE, text=True, env=env, check=False
        )
    except FileNotFoundError as error:
        raise DependencyError(
            "Install curl to download upstream assets, or supply existing paths."
        ) from error
    return result.returncode, result.stdout.strip()


def _transfer_or_raise(url: str, destination: Path, limit: int) -> None:
    code, http = _transfer(url, destination, limit)
    if code:
        raise DependencyError(f"Could not download {url} (curl {code}, HTTP {http}). Rerun setup.")


def _fetch_weight() -> None:
    target = _model_path()
    _outside_project(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name("model.safetensors.part")
    if os.path.lexists(partial) and (
        partial.is_symlink() or not partial.is_file() or partial.stat().st_nlink != 1
    ):
        raise DependencyError(f"Expected a regular private download file at {partial}")
    size = partial.stat().st_size if partial.exists() else 0
    if size > MODEL_SIZE:
        raise DependencyError(f"Oversized partial download at {partial}; move it aside to retry.")
    print(
        f"Fetching DA² weights from huggingface.co/{MODEL_REPOSITORY} ({MODEL_SIZE:,} bytes)",
        flush=True,
    )
    print(
        f"Model card: https://huggingface.co/{MODEL_REPOSITORY}/blob/{MODEL_REVISION}/README.md",
        flush=True,
    )
    url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/model.safetensors"
    if size < MODEL_SIZE:
        if size:
            print(f"Resuming at byte {size:,}", flush=True)
        code, http = _transfer(url, partial, MODEL_SIZE, resume=size > 0)
        if size and (code == 33 or http == "416"):
            print("The server declined the byte range. Restarting the download.", flush=True)
            partial.unlink()
            code, http = _transfer(url, partial, MODEL_SIZE)
        if code and http in ("401", "403"):
            raise DependencyError(
                f"Publisher returned HTTP {http}. Check access to {MODEL_REPOSITORY} on Hugging Face "
                "or supply an authorized local file with --da2-checkpoint."
            )
        if code:
            raise DependencyError(
                f"DA² download stopped (curl {code}, HTTP {http}). Rerun to resume."
            )
    try:
        _verify_weight(partial)
    except DependencyError as error:
        partial.unlink()
        raise DependencyError(
            f"{error}. Discarded the temporary file; rerun setup to retry."
        ) from error
    # Publish only a complete, verified file. An existing destination wins.
    os.link(partial, target)
    partial.unlink()


def _outside_project(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "pyproject.toml").is_file() and (
            parent / "spacefold/panorama/infer.py"
        ).is_file():
            raise DependencyError(f"Choose a cache outside the SpaceFold checkout: {resolved}")


@contextmanager
def _setup_lock(root: Path):
    import fcntl  # Setup targets macOS and Linux.

    _outside_project(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(root / ".setup.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _assets(geometry_only: bool, weights: bool, fetch: bool) -> list[str]:
    errors = []
    for source in (VOLFILL,) if geometry_only else (VOLFILL, DA2):
        try:
            if fetch and not os.path.lexists(source.location()):
                _fetch_source(source)
            print(f"Ready  {source.name} {source.revision[:12]}  {cached_source(source)}")
        except (DependencyError, OSError) as error:
            if fetch:
                raise
            errors.append(str(error))
    if weights:
        try:
            if fetch and not os.path.lexists(_model_path()):
                _fetch_weight()
                path = _model_path()  # Already hashed before publication.
            else:
                path = cached_da2_checkpoint()
            print(f"Ready  DA² weights {MODEL_SHA256[:12]}  {path}")
        except (DependencyError, OSError) as error:
            if fetch:
                raise
            errors.append(str(error))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--geometry-only", action="store_true", help="Prepare only the VolFill source."
    )
    scope.add_argument(
        "--with-da2-weights",
        action="store_true",
        help="Also fetch the official 1.38 GB DA² checkpoint.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the selected assets offline without writing files.",
    )
    args = parser.parse_args()
    try:
        root = cache_dir()
        print(f"SpaceFold dependency cache: {root}", flush=True)
        if args.check:
            errors = _assets(args.geometry_only, args.with_da2_weights, fetch=False)
            if errors:
                parser.exit(1, "\n".join(errors) + "\n")
        else:
            with _setup_lock(root):
                _assets(args.geometry_only, args.with_da2_weights, fetch=True)
    except KeyboardInterrupt:
        parser.exit(130, "\nSetup interrupted. Rerun the same command to continue.\n")
    except (DependencyError, OSError, tarfile.TarError, EOFError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

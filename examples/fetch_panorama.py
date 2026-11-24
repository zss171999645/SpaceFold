"""Download the tonemapped Empty Warehouse 01 panorama from its publisher.

Photograph by Sergej Majboroda, distributed by Poly Haven under CC0 1.0.
Asset: https://polyhaven.com/a/empty_warehouse_01
Terms: https://polyhaven.com/license
"""

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/empty_warehouse_01.jpg"
SHA256 = "c40091f113f33868fc966b3762749f48d8470da3a4267c09b152dff79e089511"
SIZE = 4_072_009


def verified_existing(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == SHA256


def fetch(destination: Path) -> Path:
    if destination.exists():
        if verified_existing(destination):
            return destination
        raise FileExistsError(f"Choose a new path or move the existing file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "SpaceFold-Panorama-Example/1.0"})
    with NamedTemporaryFile(dir=destination.parent, prefix=".warehouse-", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        digest = hashlib.sha256()
        size = 0
        with urlopen(request, timeout=45) as response, temporary.open("wb") as out:
            while block := response.read(128 * 1024):
                size += len(block)
                if size > SIZE:
                    raise ValueError("Download exceeds the pinned asset size")
                digest.update(block)
                out.write(block)
        if size != SIZE or digest.hexdigest() != SHA256:
            raise ValueError("Downloaded asset has an unexpected size or SHA-256")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not verified_existing(destination):
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the real indoor panorama used in the example."
    )
    parser.add_argument(
        "destination", type=Path, nargs="?", default=Path("examples/downloads/warehouse.jpg")
    )
    path = fetch(parser.parse_args().destination)
    print(f"{path}  (SHA-256 {SHA256})")


if __name__ == "__main__":
    main()

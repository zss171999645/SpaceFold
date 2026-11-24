"""Offline regression tests for the public setup command. Run with unittest discover."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLFILL = "a0c8e7779ccee318e7a75bb2d273290d4ffb76ed"


def setup(cache: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "spacefold.dependencies", *arguments],
        cwd=ROOT,
        env={
            **os.environ,
            "SPACEFOLD_CACHE_DIR": str(cache),
            "PATH": str(cache.parent / "no-download-tools"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


class DependencySetupTests(unittest.TestCase):
    def test_offline_check_reports_each_missing_source_without_creating_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "not-created"
            result = setup(cache, "--check")
            self.assertNotEqual(result.returncode, 0)
            output = result.stdout + result.stderr
            self.assertIn("VolFill", output)
            self.assertIn("DA-2", output)
            self.assertFalse(cache.exists())

    def test_existing_unrecognized_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            path = cache / "sources" / "VolFill" / VOLFILL
            path.mkdir(parents=True)
            existing = path / "README.md"
            original = (ROOT / "README.md").read_bytes()
            existing.write_bytes(original)
            result = setup(cache, "--geometry-only")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(path), result.stderr)
            self.assertEqual(existing.read_bytes(), original)
            self.assertFalse((path / ".git").exists())

    def test_refuses_a_managed_cache_in_the_source_repository(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            cache = Path(temp) / "cache"
            result = setup(cache, "--geometry-only")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside", result.stderr)
            self.assertFalse(cache.exists())

    def test_cache_subdirectory_cannot_redirect_an_install_into_the_repository(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory(dir=ROOT) as inside:
            cache = Path(temp)
            (cache / "sources").symlink_to(inside, target_is_directory=True)
            result = setup(cache, "--geometry-only")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside", result.stderr)
            self.assertEqual(list(Path(inside).iterdir()), [])

    def test_external_panorama_flags_remain_available_as_overrides(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "spacefold.panorama.infer", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        for flag in ("--da2-root", "--volfill-root", "--da2-checkpoint"):
            self.assertIn("[" + flag, result.stdout)

    def test_geometry_only_and_depth_weights_are_incompatible(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cache"
            result = setup(cache, "--geometry-only", "--with-da2-weights")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()

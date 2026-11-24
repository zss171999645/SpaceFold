"""Exercise panorama preprocessing with the real photograph used by the example."""

import hashlib
import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

PHOTO = Path(__file__).resolve().parents[1] / "examples/downloads/warehouse.jpg"


class PanoramaInputTests(unittest.TestCase):
    def test_accepted_numpy_scalars_are_serializable_scene_settings(self):
        import numpy as np

        from spacefold.panorama import _validate_camera_height, _validate_sampling

        height = np.float32(1.65)
        normalized_height = _validate_camera_height(height)
        self.assertIs(type(normalized_height), float)
        steps, guidance, seed = _validate_sampling(50, np.float32(3.0), 0)
        settings = dict(
            camera_height_m=normalized_height, steps=steps, guidance=guidance, seed=seed
        )
        self.assertEqual(json.loads(json.dumps(settings))["camera_height_m"], float(height))

    @unittest.skipUnless(
        find_spec("torch") and find_spec("PIL") and find_spec("safetensors"),
        "requires the panorama inference dependencies",
    )
    def test_exif_orientations_preserve_upright_rgb_and_identify_the_source_file(self):
        from PIL import Image
        from torch import equal

        from spacefold.panorama.pipeline import _image

        photo = Path(os.environ.get("SPACEFOLD_TEST_PANORAMA", PHOTO))
        if not photo.is_file():
            self.skipTest("Fetch the example photograph or set SPACEFOLD_TEST_PANORAMA")
        with Image.open(photo) as source:
            upright = source.convert("RGB").resize((512, 256))
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "upright.png"
            upright.save(base)
            expected = _image(base)[0]
            for tag, rotation in ((3, Image.Transpose.ROTATE_180), (6, Image.Transpose.ROTATE_90)):
                with self.subTest(orientation=tag):
                    stored = upright.transpose(rotation)
                    exif = stored.getexif()
                    exif[274] = tag
                    path = Path(temp) / f"orientation-{tag}.png"
                    stored.save(path, exif=exif)
                    actual = _image(path)
                    self.assertTrue(equal(actual[0], expected))
                    self.assertEqual(actual[1], hashlib.sha256(path.read_bytes()).hexdigest())

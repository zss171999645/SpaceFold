"""Opt-in device regression using a real geometry checkpoint and raw scene latent."""

import os
import unittest

import numpy as np

CHECKPOINT = os.environ.get("SPACEFOLD_TEST_GEOMETRY_WEIGHTS")
LATENT = os.environ.get("SPACEFOLD_TEST_GEOMETRY_LATENT")
TREE = os.environ.get("SPACEFOLD_TEST_GEOMETRY_TREE")


@unittest.skipUnless(CHECKPOINT and LATENT, "supply SpaceFold test geometry weights and latent")
class GeometryCudaTests(unittest.TestCase):
    def setUp(self):
        import torch

        if torch.cuda.device_count() < 2:
            self.skipTest("requires two visible CUDA devices")

    def test_narrow_layer_norm_uses_the_input_device(self):
        import torch
        from safetensors import safe_open

        from spacefold.geometry._normalization import _layer_norm_fp32

        with np.load(LATENT, allow_pickle=False) as archive:
            features = np.ascontiguousarray(archive["raw"].reshape(16, -1).T[:128])
        with safe_open(CHECKPOINT, framework="pt", device="cpu") as checkpoint:
            prefix = "decoder.nodes.3.value_head.0."
            weight = checkpoint.get_tensor(prefix + "weight")
            bias = checkpoint.get_tensor(prefix + "bias")

        with torch.inference_mode(), torch.cuda.device(0):
            # Initialize the ambient context before working with tensors on device 1.
            torch.empty(1, device="cuda:0")
            with torch.cuda.device(1):
                x = torch.from_numpy(features).to("cuda:1")
                weight, bias = weight.to(x.device), bias.to(x.device)
                expected = _layer_norm_fp32(x, weight, bias, 1e-5).cpu()

            actual = _layer_norm_fp32(x, weight, bias, 1e-5)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, x.device)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    @unittest.skipUnless(TREE, "supply a SpaceFold test geometry tree for codec inference")
    def test_encoder_and_decoder_can_use_a_nondefault_device(self):
        import torch

        from spacefold.dependencies import VOLFILL, cached_source
        from spacefold.geometry import (
            TreeBatch,
            UnifiedTUDF,
            load_geometry_checkpoint,
            register_volfill_source,
        )

        register_volfill_source(cached_source(VOLFILL))
        target = torch.device("cuda:1")
        with torch.cuda.device(0), torch.inference_mode():
            torch.empty(1, device="cuda:0")
            model, _ = load_geometry_checkpoint(CHECKPOINT, target)
            batch = TreeBatch.from_trees([UnifiedTUDF.load(TREE)]).to(target)
            with np.load(LATENT, allow_pickle=False) as archive:
                latent = torch.from_numpy(archive["raw"]).float().unsqueeze(0).to(target)
            with torch.autocast("cuda", dtype=torch.float16):
                mean, _ = model.encoder(batch)
                levels = model.decoder(latent)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(mean.device, target)
            self.assertTrue(torch.isfinite(mean).all())
            self.assertTrue(levels)
            self.assertTrue(all(level.values.device == target for level in levels))


if __name__ == "__main__":
    unittest.main()

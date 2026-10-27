"""Unit tests for native model architectures and runtimes."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from diffdetect.models import (
    Attention,
    Block,
    DistilDIRE,
    FeedForward,
    LoRA_ViT_timm,
    NativeDinoLizerRuntime,
    NativeDistilDIRERuntime,
    Wrapper5crops,
)


class NativeModelsTests(unittest.TestCase):
    def test_native_models_exported(self) -> None:
        self.assertIsNotNone(Attention)
        self.assertIsNotNone(Block)
        self.assertIsNotNone(FeedForward)
        self.assertIsNotNone(LoRA_ViT_timm)
        self.assertIsNotNone(Wrapper5crops)
        self.assertIsNotNone(DistilDIRE)
        self.assertIsNotNone(NativeDinoLizerRuntime)
        self.assertIsNotNone(NativeDistilDIRERuntime)

    def test_distildire_runtime_requires_torch(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "DistilDIRE requires torch"):
                NativeDistilDIRERuntime()

    def test_dinolizer_runtime_requires_torch_and_timm(self) -> None:
        with patch.dict("sys.modules", {"timm": None}):
            with self.assertRaisesRegex(RuntimeError, "DinoLizer requires"):
                NativeDinoLizerRuntime()

    def test_runtime_resolves_checkpoint_via_ensure_weight(self) -> None:
        mock_torch = MagicMock()
        mock_torch.device.return_value.type = "cpu"
        mock_transforms = MagicMock()
        mock_timm = MagicMock()

        modules_patch = {
            "torch": mock_torch,
            "torchvision": MagicMock(),
            "torchvision.transforms.functional": mock_transforms,
            "timm": mock_timm,
        }

        with patch.dict("sys.modules", modules_patch):
            with patch(
                "diffdetect.models.distildire.runtime.ensure_weight",
                side_effect=FileNotFoundError("not downloaded"),
            ) as mock_ensure:
                with self.assertRaises(FileNotFoundError):
                    NativeDistilDIRERuntime(device="cpu")
                mock_ensure.assert_called_with("distildire_classifier")

            with patch(
                "diffdetect.models.dinolizer.runtime.ensure_weight",
                side_effect=FileNotFoundError("not downloaded"),
            ) as mock_ensure:
                with self.assertRaises(FileNotFoundError):
                    NativeDinoLizerRuntime(device="cpu")
                mock_ensure.assert_called_with("dinolizer_bfree")


if __name__ == "__main__":
    unittest.main()

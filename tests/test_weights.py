"""Unit tests for automatic weight management and verification."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from diffdetect import (
    WEIGHT_REGISTRY,
    WeightArtifact,
    download_weight,
    ensure_weight,
    get_cache_dir,
    verify_file,
)
from diffdetect.weights import (
    calculate_file_hash,
    download_file,
    resolve_weight_artifact,
)


class WeightManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_artifact_validation(self) -> None:
        with self.assertRaises(ValueError):
            WeightArtifact("", "file.pt", ("http://example.com",), sha256="abc")
        with self.assertRaises(ValueError):
            WeightArtifact("test", "", ("http://example.com",), sha256="abc")
        with self.assertRaises(ValueError):
            WeightArtifact("test", "file.pt", (), sha256="abc")
        with self.assertRaises(ValueError):
            WeightArtifact("test", "file.pt", ("http://example.com",))

    def test_get_cache_dir_with_and_without_env(self) -> None:
        custom_cache = self.temp_path / "custom_cache"
        with patch.dict(os.environ, {"DIFFDETECT_CACHE_DIR": str(custom_cache)}):
            cache_dir = get_cache_dir()
            self.assertEqual(cache_dir, custom_cache)
            self.assertTrue(cache_dir.is_dir())

        with patch.dict(os.environ, {"DIFFDETECT_CACHE_DIR": ""}):
            default_dir = get_cache_dir()
            self.assertTrue(str(default_dir).endswith(".cache/diffdetect/models"))

    def test_hash_calculation_and_file_verification(self) -> None:
        test_file = self.temp_path / "sample.bin"
        content = b"DiffDetect weight test data 12345"
        test_file.write_bytes(content)

        expected_sha256 = hashlib.sha256(content).hexdigest()
        expected_md5 = hashlib.md5(content).hexdigest()

        self.assertEqual(calculate_file_hash(test_file, "sha256"), expected_sha256)
        self.assertEqual(calculate_file_hash(test_file, "md5"), expected_md5)

        self.assertTrue(
            verify_file(
                test_file,
                sha256=expected_sha256,
                md5=expected_md5,
                size=len(content),
            )
        )
        # Size mismatch
        self.assertFalse(verify_file(test_file, size=999))
        # Digest mismatch
        self.assertFalse(verify_file(test_file, sha256="0" * 64))
        self.assertFalse(verify_file(test_file, md5="0" * 32))
        # Nonexistent file
        self.assertFalse(verify_file(self.temp_path / "nonexistent.bin"))

    def test_resolve_artifact_canonical_and_aliases(self) -> None:
        distildire = resolve_weight_artifact("distildire_classifier")
        self.assertEqual(distildire.name, "distildire_classifier")
        self.assertIn("imagenet-distil-dire-11e.pth", distildire.filename)

        dinolizer = resolve_weight_artifact("dinolizer")
        self.assertEqual(dinolizer.name, "dinolizer_bfree")

        dinolizer_checkpoint = resolve_weight_artifact("dinolizer_checkpoint")
        self.assertEqual(dinolizer_checkpoint.name, "dinolizer_bfree")

        with self.assertRaises(KeyError):
            resolve_weight_artifact("unknown_model_name")

    def test_download_file_success(self) -> None:
        content = b"fake downloaded model weights"
        sha256 = hashlib.sha256(content).hexdigest()
        dest = self.temp_path / "model.pth"

        mock_response = io.BytesIO(content)
        mock_response.__enter__ = lambda s: s  # type: ignore[attr-defined]
        mock_response.__exit__ = lambda s, *args: None  # type: ignore[attr-defined]

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = download_file("http://fake.url/model.pth", dest, sha256=sha256)
            self.assertEqual(result, dest)
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), content)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_download_file_hash_mismatch_cleans_up(self) -> None:
        content = b"corrupted data"
        dest = self.temp_path / "corrupted.pth"

        mock_response = io.BytesIO(content)
        mock_response.__enter__ = lambda s: s  # type: ignore[attr-defined]
        mock_response.__exit__ = lambda s, *args: None  # type: ignore[attr-defined]

        with patch("urllib.request.urlopen", return_value=mock_response):
            with self.assertRaises(ValueError):
                download_file(
                    "http://fake.url/corrupted.pth",
                    dest,
                    sha256="e" * 64,
                )
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_download_weight_reuses_valid_cache(self) -> None:
        cached_file = self.temp_path / "imagenet-distil-dire-11e.pth"
        artifact = WEIGHT_REGISTRY["distildire_classifier"]
        # Write dummy file with invalid hash
        cached_file.write_bytes(b"dummy")

        with patch(
            "diffdetect.weights.verify_file",
            return_value=True,
        ) as mock_verify, patch("diffdetect.weights.download_file") as mock_download:
            path = download_weight(
                "distildire_classifier",
                cache_dir=self.temp_path,
                force=False,
            )
            self.assertEqual(path, cached_file)
            mock_verify.assert_called_once()
            mock_download.assert_not_called()

    def test_ensure_weight_with_existing_file_and_registered_name(self) -> None:
        existing = self.temp_path / "local_weights.pt"
        existing.write_bytes(b"local weights")

        resolved_existing = ensure_weight(existing)
        self.assertEqual(resolved_existing, existing.resolve())

        with patch(
            "diffdetect.weights.download_weight",
            return_value=existing,
        ) as mock_download:
            resolved_registered = ensure_weight(
                "distildire_classifier",
                cache_dir=self.temp_path,
            )
            self.assertEqual(resolved_registered, existing)
            mock_download.assert_called_once_with(
                "distildire_classifier",
                cache_dir=self.temp_path,
            )

        with self.assertRaises(FileNotFoundError):
            ensure_weight("nonexistent_local_file_path.pt")


if __name__ == "__main__":
    unittest.main()

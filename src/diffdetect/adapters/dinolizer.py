"""Adapter boundary for the external DinoLizer localization detector."""

from __future__ import annotations

import gc
import importlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from pathlib import Path
from threading import Lock
from typing import Protocol

from PIL import Image

from ..detector import BaseDetector, DetectionResult, DetectionTarget


DEFAULT_THRESHOLD = 0.5946570634841919
WINDOW_SIZE = 504
STRIDE = 128
PIXEL_THRESHOLD = 0.5
TARGET_RESIZE_AREA = 1016 * 1016
INFERENCE_BATCH_SIZE = 50
_EXTERNAL_MODULE_ROOTS = ("networks", "utils")
_EXTERNAL_IMPORT_LOCK = Lock()


@dataclass(frozen=True)
class DinoLizerInference:
    """Detector-specific output produced by a DinoLizer runtime."""

    score: float
    localization_map: Image.Image
    marked_area: float
    processed_size: tuple[int, int]
    window_count: int


class DinoLizerRuntime(Protocol):
    """Small inference boundary used by the adapter."""

    def infer(self, image: Image.Image) -> DinoLizerInference:
        """Return calibrated image evidence and an original-size probability map."""


class DinoLizerDetector(BaseDetector):
    """Expose DinoLizer edited-image evidence through the common detector API.

    The official source tree and checkpoint remain external dependencies.
    Their modules and weights are loaded only when the first image is
    predicted. A custom runtime may be injected for testing or integration.
    """

    name = "dinolizer"
    target = DetectionTarget.EDITED

    def __init__(
        self,
        *,
        repository_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cuda",
        runtime: DinoLizerRuntime | None = None,
    ) -> None:
        self.threshold = _probability(threshold, "threshold")

        if runtime is not None:
            if not callable(getattr(runtime, "infer", None)):
                raise TypeError("runtime must provide a callable infer(image) method")
            if repository_path is not None or checkpoint_path is not None:
                raise ValueError("model paths cannot be combined with a custom runtime")
            self._runtime = runtime
            self._runtime_config: tuple[Path, Path, str] | None = None
            return

        missing = [
            name
            for name, value in (
                ("repository_path", repository_path),
                ("checkpoint_path", checkpoint_path),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"missing DinoLizer configuration: {', '.join(missing)}")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")

        assert repository_path is not None
        assert checkpoint_path is not None
        repository = _directory(repository_path, "repository_path")
        checkpoint = _file(checkpoint_path, "checkpoint_path")

        self._runtime = None
        self._runtime_config = (repository, checkpoint, device.strip())

    def predict(self, image: Image.Image) -> DetectionResult:
        """Calculate DinoLizer evidence for one image."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")

        rgb_image = image.convert("RGB")
        inference = self._get_runtime().infer(rgb_image)
        if not isinstance(inference, DinoLizerInference):
            raise TypeError("runtime must return DinoLizerInference")

        score = _probability(inference.score, "score")
        marked_area = _probability(inference.marked_area, "marked_area")
        processed_size = _size(inference.processed_size, "processed_size")
        window_count = _positive_integer(inference.window_count, "window_count")
        localization_map = _localization_map(inference.localization_map, image.size)

        return DetectionResult(
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            localization_map=localization_map,
            metadata={
                "score_name": "p99_probability",
                "marked_area": marked_area,
                "original_size": image.size,
                "processed_size": processed_size,
                "window_count": window_count,
                "window_size": WINDOW_SIZE,
                "stride": STRIDE,
                "pixel_threshold": PIXEL_THRESHOLD,
            },
        )

    def _get_runtime(self) -> DinoLizerRuntime:
        if self._runtime is None:
            if self._runtime_config is None:
                raise RuntimeError("DinoLizer runtime configuration is unavailable")
            self._runtime = _TorchDinoLizerRuntime(*self._runtime_config)
        return self._runtime


class _TorchDinoLizerRuntime:
    """Load and run the official DinoLizer implementation with PyTorch."""

    def __init__(
        self,
        repository_path: Path,
        checkpoint_path: Path,
        device: str,
    ) -> None:
        try:
            numpy = importlib.import_module("numpy")
            timm = importlib.import_module("timm")
            torch = importlib.import_module("torch")
            transforms = importlib.import_module("torchvision.transforms.functional")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DinoLizer requires numpy, timm, torch, and torchvision; install "
                "the documented external environment before running this adapter"
            ) from exc

        try:
            lora_module, wrapper_module = _import_official_modules(repository_path)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"could not import DinoLizer from repository: {repository_path}"
            ) from exc

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        backbone = timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            num_classes=1,
            pretrained=True,
        )
        model = wrapper_module.Wrapper5crops(
            backbone,
            WINDOW_SIZE,
            return_features=True,
        )
        model.model = lora_module.LoRA_ViT_timm(
            vit_model=model.model,
            r=64,
            alpha=64,
            num_classes=0,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        try:
            checkpoint_weights = checkpoint["state_dict"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid DinoLizer checkpoint") from exc
        if not isinstance(checkpoint_weights, Mapping):
            raise ValueError("invalid DinoLizer checkpoint state dictionary")

        state_dict = {}
        for name, value in checkpoint_weights.items():
            if not isinstance(name, str):
                raise ValueError("invalid DinoLizer checkpoint parameter name")
            normalized_name = name.removeprefix("net.").removeprefix("model.")
            if "pos_weight" not in normalized_name:
                state_dict[normalized_name] = value

        model.load_state_dict(state_dict)
        model = model.half().to(torch_device).eval()

        del backbone, checkpoint, checkpoint_weights, state_dict
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        self._numpy = numpy
        self._torch = torch
        self._transforms = transforms
        self._device = torch_device
        self._model = model

    def infer(self, image: Image.Image) -> DinoLizerInference:
        numpy = self._numpy
        torch = self._torch

        original_size = image.size
        processed_size = _processed_size(original_size)
        if processed_size != original_size:
            image = image.resize(processed_size, Image.Resampling.BICUBIC)

        coordinates = _window_coordinates(processed_size)
        width, height = processed_size
        logit_sums = numpy.zeros((height, width, 3), dtype=numpy.float32)
        weight_matrix = numpy.zeros((height, width), dtype=numpy.float32)

        for start in range(0, len(coordinates), INFERENCE_BATCH_SIZE):
            batch_coordinates = coordinates[start : start + INFERENCE_BATCH_SIZE]
            batch = torch.stack(
                [self._transform(image.crop(coordinate)) for coordinate in batch_coordinates]
            ).to(device=self._device, dtype=torch.float16)

            with torch.inference_mode():
                _, features = self._model(batch)
                patch_logits = self._model.cls_head(features[:, 5:, :])
                patch_logits = patch_logits.reshape(-1, 36, 36, 3).permute(0, 3, 1, 2)
                patch_logits = torch.nn.functional.interpolate(
                    patch_logits,
                    size=(WINDOW_SIZE, WINDOW_SIZE),
                    mode="bicubic",
                    align_corners=False,
                )
                patch_logits = patch_logits.permute(0, 2, 3, 1).cpu().numpy()

            for coordinate, window_logits in zip(batch_coordinates, patch_logits):
                left, top, right, bottom = coordinate
                logit_sums[top:bottom, left:right, :] += window_logits
                weight_matrix[top:bottom, left:right] += 1.0

        if numpy.any(weight_matrix == 0.0):
            raise RuntimeError("DinoLizer windows did not cover the processed image")

        averaged_logits = logit_sums / weight_matrix[:, :, numpy.newaxis]
        shifted_logits = averaged_logits - numpy.max(
            averaged_logits,
            axis=2,
            keepdims=True,
        )
        exponentials = numpy.exp(shifted_logits)
        probabilities = exponentials / numpy.sum(
            exponentials,
            axis=2,
            keepdims=True,
        )
        processed_map = probabilities[:, :, 2].astype(numpy.float32, copy=False)

        score = float(numpy.percentile(processed_map, 99))
        marked_area = float(numpy.mean(processed_map > PIXEL_THRESHOLD))
        localization_map = self._restore_original_size(processed_map, original_size)

        return DinoLizerInference(
            score=score,
            localization_map=localization_map,
            marked_area=marked_area,
            processed_size=processed_size,
            window_count=len(coordinates),
        )

    def _transform(self, image: Image.Image):
        tensor = self._transforms.to_tensor(image)
        return self._transforms.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def _restore_original_size(
        self,
        probability_map,
        original_size: tuple[int, int],
    ) -> Image.Image:
        numpy = self._numpy
        localization_map = Image.fromarray(probability_map)
        if localization_map.size != original_size:
            localization_map = localization_map.resize(
                original_size,
                Image.Resampling.BICUBIC,
            )
        restored = numpy.asarray(localization_map, dtype=numpy.float32)
        restored = numpy.clip(restored, 0.0, 1.0)
        return Image.fromarray(restored.astype(numpy.float32, copy=False))


def _import_official_modules(repository_path: Path):
    """Import DinoLizer without retaining its generic top-level packages.

    DinoLizer and other external detectors use names such as ``networks`` and
    ``utils``. Temporarily isolating those names lets their model objects live
    together in one DiffDetect process without one repository shadowing the
    other in ``sys.modules``.
    """

    repository = str(repository_path)
    with _EXTERNAL_IMPORT_LOCK:
        previous_modules = {
            name: module
            for name, module in sys.modules.items()
            if _is_external_module(name)
        }
        for name in previous_modules:
            del sys.modules[name]

        sys.path.insert(0, repository)
        try:
            lora_module = importlib.import_module("networks.LORA")
            wrapper_module = importlib.import_module("networks.wrapper5crops")
        finally:
            for name in tuple(sys.modules):
                if _is_external_module(name):
                    del sys.modules[name]
            sys.modules.update(previous_modules)
            if sys.path and sys.path[0] == repository:
                del sys.path[0]
            else:
                sys.path.remove(repository)

    return lora_module, wrapper_module


def _is_external_module(name: str) -> bool:
    return any(
        name == root or name.startswith(f"{root}.")
        for root in _EXTERNAL_MODULE_ROOTS
    )


def _processed_size(original_size: tuple[int, int]) -> tuple[int, int]:
    width, height = _size(original_size, "original_size")
    if width >= WINDOW_SIZE and height >= WINDOW_SIZE:
        return original_size

    scale = sqrt(TARGET_RESIZE_AREA / float(width * height))
    processed_width = max(1, int(round(width * scale)))
    processed_height = max(1, int(round(height * scale)))

    if processed_width < WINDOW_SIZE or processed_height < WINDOW_SIZE:
        minimum_scale = max(
            WINDOW_SIZE / processed_width,
            WINDOW_SIZE / processed_height,
        )
        processed_width = int(round(processed_width * minimum_scale))
        processed_height = int(round(processed_height * minimum_scale))

    return processed_width, processed_height


def _window_coordinates(
    image_size: tuple[int, int],
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = _size(image_size, "image_size")
    horizontal = _window_starts(width, window_size, stride)
    vertical = _window_starts(height, window_size, stride)
    return tuple(
        (left, top, left + window_size, top + window_size)
        for left in horizontal
        for top in vertical
    )


def _window_starts(length: int, window_size: int, stride: int) -> tuple[int, ...]:
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if type(stride) is not int or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if type(length) is not int or length < window_size:
        raise ValueError("image dimensions must be at least the window size")

    final_start = length - window_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _size(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _localization_map(value: object, original_size: tuple[int, int]) -> Image.Image:
    if not isinstance(value, Image.Image):
        raise TypeError("localization_map must be a PIL image")
    if value.mode != "F":
        raise ValueError("localization_map must use floating-point mode F")
    if value.size != original_size:
        raise ValueError("localization_map must match the original image size")

    if any(
        not isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in value.getdata()
    ):
        raise ValueError("localization_map values must be finite and between 0 and 1")
    return value


def _directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")
    return path.resolve()


def _file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path.resolve()

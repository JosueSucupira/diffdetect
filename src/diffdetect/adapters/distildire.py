"""Adapter for the external DistilDIRE ImageNet detector."""

from __future__ import annotations

import gc
import importlib
import sys
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Protocol

from PIL import Image

from ..detector import BaseDetector, DetectionResult, DetectionTarget


class DistilDIRERuntime(Protocol):
    """Small inference boundary used by the adapter."""

    def score(self, image: Image.Image) -> float:
        """Return the sigmoid score for synthetic-image evidence."""


class DistilDIREDetector(BaseDetector):
    """Expose DistilDIRE synthetic evidence through the common detector API.

    The official DistilDIRE repository is loaded at runtime and remains an
    external dependency. Model weights are never downloaded by this adapter.
    A threshold is required because the official 0.5 threshold did not
    generalize to the validation images used by this project.
    """

    name = "distildire"
    target = DetectionTarget.SYNTHETIC

    def __init__(
        self,
        *,
        threshold: float,
        repository_path: str | Path | None = None,
        classifier_weights: str | Path | None = None,
        adm_weights: str | Path | None = None,
        device: str = "cuda",
        runtime: DistilDIRERuntime | None = None,
    ) -> None:
        self.threshold = _probability(threshold, "threshold")

        if runtime is not None:
            if not callable(getattr(runtime, "score", None)):
                raise TypeError("runtime must provide a callable score(image) method")
            if any(
                value is not None
                for value in (repository_path, classifier_weights, adm_weights)
            ):
                raise ValueError("model paths cannot be combined with a custom runtime")
            self._runtime = runtime
            self._runtime_config: tuple[Path, Path, Path, str] | None = None
            return

        missing = [
            name
            for name, value in (
                ("repository_path", repository_path),
                ("classifier_weights", classifier_weights),
                ("adm_weights", adm_weights),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"missing DistilDIRE configuration: {', '.join(missing)}")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")

        assert repository_path is not None
        assert classifier_weights is not None
        assert adm_weights is not None
        repository = _directory(repository_path, "repository_path")
        classifier = _file(classifier_weights, "classifier_weights")
        adm = _file(adm_weights, "adm_weights")

        self._runtime = None
        self._runtime_config = (repository, classifier, adm, device.strip())

    def predict(self, image: Image.Image) -> DetectionResult:
        """Calculate DistilDIRE evidence for one image."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")

        score = _probability(self._get_runtime().score(image.convert("RGB")), "score")
        return DetectionResult(
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
        )

    def _get_runtime(self) -> DistilDIRERuntime:
        if self._runtime is None:
            if self._runtime_config is None:
                raise RuntimeError("DistilDIRE runtime configuration is unavailable")
            self._runtime = _TorchDistilDIRERuntime(*self._runtime_config)
        return self._runtime


class _TorchDistilDIRERuntime:
    """Load and run the official DistilDIRE implementation with PyTorch."""

    def __init__(
        self,
        repository_path: Path,
        classifier_weights: Path,
        adm_weights: Path,
        device: str,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            transforms = importlib.import_module("torchvision.transforms.functional")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DistilDIRE requires torch and torchvision; install the documented "
                "optional environment before running this adapter"
            ) from exc

        repository = str(repository_path)
        if repository not in sys.path:
            sys.path.insert(0, repository)

        try:
            compute = importlib.import_module("guided_diffusion.compute_dire_eps")
            distill_model = importlib.import_module("networks.distill_model")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"could not import DistilDIRE from repository: {repository_path}"
            ) from exc

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        detector = distill_model.DistilDIRE(str(torch_device))
        checkpoint = torch.load(
            classifier_weights,
            map_location="cpu",
            weights_only=True,
        )
        try:
            checkpoint_weights = checkpoint["model"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid DistilDIRE classifier checkpoint") from exc

        state_dict = {
            name.removeprefix("module."): value
            for name, value in checkpoint_weights.items()
        }
        detector.load_state_dict(state_dict, strict=True)
        detector = detector.to(torch_device).eval()

        adm_args = compute.create_dicts_for_static_init()
        adm_args["timestep_respacing"] = "ddim20"
        adm_config = compute.dict_parse(
            adm_args,
            compute.model_and_diffusion_defaults().keys(),
        )
        adm_model, diffusion = compute.create_model_and_diffusion(**adm_config)
        adm_state_dict = torch.load(
            adm_weights,
            map_location="cpu",
            weights_only=True,
        )
        adm_model.load_state_dict(adm_state_dict, strict=True)
        adm_model = adm_model.to(torch_device).float().eval()

        del checkpoint, checkpoint_weights, state_dict, adm_state_dict
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        self._torch = torch
        self._transforms = transforms
        self._device = torch_device
        self._detector = detector
        self._adm_model = adm_model
        self._diffusion = diffusion
        self._adm_args = adm_args
        self._noise = compute.dire_get_first_step_noise

    def score(self, image: Image.Image) -> float:
        torch = self._torch
        transforms = self._transforms

        tensor = transforms.to_tensor(image) * 2.0 - 1.0
        tensor = transforms.resize(tensor, 256, antialias=True)
        tensor = transforms.center_crop(tensor, [256, 256])
        tensor = tensor.unsqueeze(0).to(
            device=self._device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            noise = self._noise(
                tensor,
                self._adm_model,
                self._diffusion,
                self._adm_args,
                str(self._device),
            )
            output = self._detector(tensor, noise)

        if not isinstance(output, dict) or "logit" not in output:
            raise RuntimeError("DistilDIRE output does not contain a logit")
        logit = output["logit"]
        if logit.numel() != 1:
            raise RuntimeError("DistilDIRE returned more than one score for one image")
        return float(logit.sigmoid().item())


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


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

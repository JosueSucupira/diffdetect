"""Native PyTorch runtime for DistilDIRE without external repository dependencies."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PIL import Image

from ...weights import ensure_weight
from .classifier import DistilDIRE

if TYPE_CHECKING:
    import torch


class NativeDistilDIRERuntime:
    """Load and execute DistilDIRE natively using PyTorch."""

    def __init__(
        self,
        classifier_weights: str | Path | None = None,
        adm_weights: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            import torchvision.transforms.functional as transforms
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DistilDIRE requires torch and torchvision; "
                "install the optional models environment: pip install '.[models]'"
            ) from exc

        if classifier_weights is None:
            classifier_file = ensure_weight("distildire_classifier")
        else:
            classifier_file = Path(classifier_weights).expanduser().resolve()
            if not classifier_file.is_file():
                raise FileNotFoundError(f"classifier weights not found: {classifier_file}")

        if adm_weights is None:
            adm_file = ensure_weight("distildire_adm")
        else:
            adm_file = Path(adm_weights).expanduser().resolve()
            if not adm_file.is_file():
                raise FileNotFoundError(f"ADM weights not found: {adm_file}")

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        # 1. Build & Load DistilDIRE student classifier
        detector = DistilDIRE(str(torch_device))
        checkpoint = torch.load(
            classifier_file,
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

        # 2. Build & Load ADM UNet and SpacedDiffusion
        from .adm.gaussian_diffusion import (
            LossType,
            ModelMeanType,
            ModelVarType,
            get_named_beta_schedule,
        )
        from .adm.respace import SpacedDiffusion, space_timesteps
        from .adm.unet import UNetModel

        adm_model = UNetModel(
            image_size=256,
            in_channels=3,
            model_channels=256,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=(8, 16, 32),
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        )
        adm_state = torch.load(
            adm_file,
            map_location="cpu",
            weights_only=True,
        )
        adm_model.load_state_dict(adm_state, strict=True)
        adm_model = adm_model.to(torch_device).float().eval()

        betas = get_named_beta_schedule("linear", 1000)
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, "ddim20"),
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.LEARNED_RANGE,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        del checkpoint, checkpoint_weights, state_dict, adm_state
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        self._torch = torch
        self._transforms = transforms
        self._device = torch_device
        self._detector = detector
        self._adm_model = adm_model
        self._diffusion = diffusion

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

        t_zero = torch.zeros(tensor.shape[0], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            noise = self._diffusion.ddim_reverse_sample_only_eps(
                self._adm_model,
                x=tensor,
                t=t_zero,
                clip_denoised=True,
                model_kwargs={},
                eta=0.0,
            )
            output = self._detector(tensor, noise)

        if not isinstance(output, dict) or "logit" not in output:
            raise RuntimeError("DistilDIRE output does not contain a logit")
        logit = output["logit"]
        if logit.numel() != 1:
            raise RuntimeError("DistilDIRE returned more than one score for one image")
        return float(logit.sigmoid().item())

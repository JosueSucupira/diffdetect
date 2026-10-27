"""Automatic weight management and verification for DiffDetect models."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR_NAME = ".cache/diffdetect/models"
_USER_AGENT = "DiffDetect/0.1.0"
_STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class WeightArtifact:
    """Specification of a downloadable model weight artifact."""

    name: str
    filename: str
    urls: tuple[str, ...]
    sha256: str | None = None
    md5: str | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if not self.urls:
            raise ValueError("urls must contain at least one download URL")
        if not self.sha256 and not self.md5:
            raise ValueError("at least one of sha256 or md5 digest must be specified")


WEIGHT_REGISTRY: dict[str, WeightArtifact] = {
    "distildire_classifier": WeightArtifact(
        name="distildire_classifier",
        filename="imagenet-distil-dire-11e.pth",
        urls=(
            "https://huggingface.co/yevvonlim/distildire/resolve/1318c17f77ec70153365a3cd62d885a87c498e7d/imagenet-distil-dire-11e.pth?download=true",
        ),
        sha256="e6b76a13ae49eb83d39fb9b1f7de86bf9f63d1dddf0225ca7ad03690e6bc53bc",
    ),
    "distildire_adm": WeightArtifact(
        name="distildire_adm",
        filename="256x256-adm.pt",
        urls=(
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
        ),
        md5="fd9dd2335b8736d521de0aed54bd90ca",
    ),
    "dinolizer_bfree": WeightArtifact(
        name="dinolizer_bfree",
        filename="epoch=36_val_loss=0.2130.ckpt",
        urls=(
            "https://nextcloud.univ-lille.fr/index.php/s/HS6CLT73DZpZEyq/download",
        ),
        sha256="96cb26f2536919d67b75a5aff195b683a1d8590fe596e6f1260ab7950883e2a2",
        size=386800693,
    ),
}

_ALIASES: dict[str, str] = {
    "dinolizer": "dinolizer_bfree",
    "dinolizer_checkpoint": "dinolizer_bfree",
}


def get_cache_dir() -> Path:
    """Return the base cache directory for models, creating it if needed."""
    cache_env = os.environ.get("DIFFDETECT_CACHE_DIR")
    if cache_env and cache_env.strip():
        cache_dir = Path(cache_env.strip()).expanduser()
    else:
        cache_dir = Path.home() / DEFAULT_CACHE_DIR_NAME

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def calculate_file_hash(
    path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = _STREAM_CHUNK_SIZE,
) -> str:
    """Calculate the cryptographic hash digest of a file in streaming chunks."""
    file_path = Path(path)
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(
    path: str | Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
    size: int | None = None,
) -> bool:
    """Check if a file matches the expected size and hash digests."""
    file_path = Path(path)
    if not file_path.is_file():
        return False

    if size is not None:
        try:
            if file_path.stat().st_size != size:
                return False
        except OSError:
            return False

    if sha256 is not None:
        if calculate_file_hash(file_path, "sha256").lower() != sha256.lower():
            return False

    if md5 is not None:
        if calculate_file_hash(file_path, "md5").lower() != md5.lower():
            return False

    return True


def resolve_weight_artifact(name: str) -> WeightArtifact:
    """Resolve an artifact specification by canonical name or alias."""
    canonical_name = _ALIASES.get(name, name)
    if canonical_name not in WEIGHT_REGISTRY:
        available = sorted(set(WEIGHT_REGISTRY.keys()) | set(_ALIASES.keys()))
        raise KeyError(
            f"unknown weight artifact '{name}'. Available artifacts: {', '.join(available)}"
        )
    return WEIGHT_REGISTRY[canonical_name]


def download_file(
    url: str,
    destination: str | Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
    size: int | None = None,
    chunk_size: int = _STREAM_CHUNK_SIZE,
) -> Path:
    """Download a remote URL to destination, verifying digests atomically."""
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, part_path.open("wb") as out:
            while chunk := response.read(chunk_size):
                out.write(chunk)
    except Exception as exc:
        if part_path.exists():
            part_path.unlink()
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    if not verify_file(part_path, sha256=sha256, md5=md5, size=size):
        if part_path.exists():
            part_path.unlink()
        raise ValueError(
            f"downloaded artifact from {url} failed digest verification for {dest_path.name}"
        )

    part_path.replace(dest_path)
    return dest_path


def download_weight(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure a registered model weight artifact is downloaded and valid in cache."""
    artifact = resolve_weight_artifact(name)
    base_dir = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    destination = base_dir / artifact.filename

    if not force and destination.exists():
        if verify_file(
            destination,
            sha256=artifact.sha256,
            md5=artifact.md5,
            size=artifact.size,
        ):
            return destination

    errors: list[str] = []
    for url in artifact.urls:
        try:
            return download_file(
                url,
                destination,
                sha256=artifact.sha256,
                md5=artifact.md5,
                size=artifact.size,
            )
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        f"could not download weight artifact '{artifact.name}' from any mirror:\n"
        + "\n".join(errors)
    )


def ensure_weight(
    name_or_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a path to a valid local file or download a registered model weight.

    If name_or_path matches an existing file, its Path is returned directly.
    Otherwise, it is treated as a registered artifact name and fetched into cache.
    """
    path_candidate = Path(name_or_path)
    if path_candidate.is_file():
        return path_candidate.resolve()

    if isinstance(name_or_path, str):
        try:
            return download_weight(name_or_path, cache_dir=cache_dir)
        except KeyError:
            pass

    raise FileNotFoundError(
        f"weight '{name_or_path}' is neither an existing file nor a known artifact name"
    )

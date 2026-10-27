"""Public API for DiffDetect detector adapters."""

from .adapters import DinoLizerDetector, DinoLizerInference, DistilDIREDetector
from .classification import ClassificationResult, DecisionReason, ImageClass, classify
from .detector import BaseDetector, DetectionResult, DetectionTarget
from .hierarchical import (
    HIERARCHICAL_FEATURES,
    POLICY_SCHEMA_VERSION,
    HierarchicalClassificationResult,
    HierarchicalDecisionReason,
    HierarchicalPolicy,
    LogisticPolicyModel,
    classify_hierarchical,
    load_hierarchical_policy,
)
from .registry import DetectorRegistry
from .runner import DetectorRun, DetectorRunner
from .weights import (
    WEIGHT_REGISTRY,
    WeightArtifact,
    download_weight,
    ensure_weight,
    get_cache_dir,
    verify_file,
)

__all__ = [
    "BaseDetector",
    "ClassificationResult",
    "DecisionReason",
    "DetectionResult",
    "DetectionTarget",
    "DinoLizerDetector",
    "DinoLizerInference",
    "DistilDIREDetector",
    "DetectorRegistry",
    "DetectorRun",
    "DetectorRunner",
    "ImageClass",
    "HIERARCHICAL_FEATURES",
    "POLICY_SCHEMA_VERSION",
    "HierarchicalClassificationResult",
    "HierarchicalDecisionReason",
    "HierarchicalPolicy",
    "LogisticPolicyModel",
    "WEIGHT_REGISTRY",
    "WeightArtifact",
    "classify",
    "classify_hierarchical",
    "download_weight",
    "ensure_weight",
    "get_cache_dir",
    "load_hierarchical_policy",
    "verify_file",
]

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
    "classify",
    "classify_hierarchical",
    "load_hierarchical_policy",
]

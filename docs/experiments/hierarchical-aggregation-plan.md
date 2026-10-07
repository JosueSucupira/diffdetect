# Hierarchical aggregation experiment plan

This experiment will implement and evaluate the two-level aggregation proposed
for DiffDetect. It follows the negative Boolean baseline documented in
[`three-class-validation.md`](three-class-validation.md), where both adapters
completed successfully but the final rule produced 59.38% conflicts and only
6.64% end-to-end accuracy.

The baseline `classify` function will remain unchanged. A separate,
configuration-driven hierarchical policy will be calibrated, serialized, and
evaluated so the two approaches can be reproduced side by side.

## Research question

Can the existing DistilDIRE and DinoLizer evidence be combined hierarchically
to distinguish `real`, `synthetic`, and `edited` images with useful coverage,
without comparing uncalibrated raw scores directly?

The intended hierarchy is:

1. determine whether an image is real or contains artificial evidence;
2. only for artificial images, determine whether the evidence is more
   consistent with full synthesis or localized editing;
3. abstain when either level lacks sufficient confidence.

This structure implements the aggregation concept described in TCC I more
faithfully than the Boolean baseline, which treats simultaneous detector
positives as an unresolved conflict.

## Scope and frozen components

The first iteration will use only the two already integrated adapters. Adding
a third model before measuring this hierarchy would prevent isolating the
effect of the aggregation change.

The following components remain frozen:

- `DistilDIREDetector`, upstream revision, checkpoints, preprocessing, and raw
  score definition;
- `DinoLizerDetector`, upstream revision, checkpoint, preprocessing, raw score,
  and marked-area definition;
- the existing `classify` function and its tests;
- the 768-row three-class baseline result and its hashes.

The adapters' existing Boolean thresholds are not inputs to the hierarchical
models. The hierarchy consumes detector scores and metadata, calibrates each
feature inside the aggregation policy, and preserves the original Boolean
outputs only for audit and baseline comparison.

## Input features

Each image supplies three bounded numeric features:

| Feature | Source | Meaning |
| --- | --- | --- |
| `distildire_score` | DistilDIRE | Detector-specific synthetic score |
| `dinolizer_score` | DinoLizer | 99th percentile of the class-2 probability map |
| `dinolizer_marked_area` | DinoLizer metadata | Proportion of pixels whose edited probability is at least 0.5 |

Raw scores from different detectors will not be directly compared. Each
feature will be standardized using a mean and standard deviation learned only
from the calibration manifest. The fitted means, deviations, coefficients,
intercepts, and confidence thresholds will be stored in a versioned JSON
policy artifact.

Missing, non-finite, or out-of-range required features produce an inconclusive
decision. A detector execution error remains a detector error rather than a
low-confidence prediction.

## Primary hierarchical model

The primary policy contains two L2-regularized logistic regressions with the
same three standardized inputs.

### Level 1: authenticity

- Negative class: `real`
- Positive class: `artificial`, combining `synthetic` and `edited`
- Model: logistic regression with `C=1.0`, `class_weight="balanced"`,
  `solver="lbfgs"`, and at most 1,000 iterations
- Output: `p_artificial`

For a frozen confidence threshold `c1`, where `0.5 <= c1 < 1.0`:

- `p_artificial <= 1 - c1` returns `real`;
- `p_artificial >= c1` proceeds to level 2;
- intermediate values are inconclusive.

### Level 2: modality

- Negative class: `edited`
- Positive class: `synthetic`
- Training rows: only calibration images labeled `edited` or `synthetic`
- Model: the same fixed logistic-regression configuration
- Output: `p_synthetic_given_artificial`

For a frozen confidence threshold `c2`:

- `p_synthetic_given_artificial <= 1 - c2` returns `edited`;
- `p_synthetic_given_artificial >= c2` returns `synthetic`;
- intermediate values are inconclusive.

The second-level probability is interpreted only after level 1 accepts the
artificial branch. Neither output is a universal probability outside the
calibration domain.

## Calibration corpus

Calibration will contain 768 images, balanced with 256 per class.

### Real and edited classes

Use the 256 real and 256 edited images from the existing CocoGlide
**calibration** split. Its COCO source identifiers do not overlap the
CocoGlide validation split used in the Boolean baseline.

- Dataset revision: `275f045df7caa2544dd28de7fa86b044ab661bd0`
- Split-manifest SHA-256:
  `8311d0e96ccb83bfbfdb2ee981d604515c921701c6b4e7210c0646fc6217e21c`
- Existing DinoLizer calibration result:
  [`results/dinolizer-cocoglide-calibration.csv`](results/dinolizer-cocoglide-calibration.csv)

Both detectors will be run through the current experiment runner on the frozen
images. The existing DinoLizer CSV may be used for consistency checks but will
not be joined silently with a new partial run.

### Synthetic class

Select 256 new DiffusionDB images without inspecting detector output. Use the
protocol identifier `diffdetect-hierarchical-calibration-v1`:

1. its SHA-256 is
   `c3db1e891535a757243d643e9ab864e6c9f8b9ea7beb056cf86ec5608770ecd2`;
2. interpreting the first 16 hexadecimal digits, taking modulo 2,000, and
   adding one selects archive part `001800`;
3. apply the same finite NSFW-score, maximum `0.1`, valid-dimension, and
   existing-file filters as the three-class baseline;
4. rank eligible filenames by SHA-256 of
   `diffdetect-hierarchical-calibration-v1:{image_name}` and take the first
   256;
5. continue cyclically through later parts only if the selected archive does
   not supply 256 eligible files, recording every part used.

The current validation selected DiffusionDB part `001079`; the new calibration
selection must have no image identifier or input hash in common with it.

## Cross-validation and confidence thresholds

The policy structure and logistic-regression hyperparameters are fixed above.
Only the fitted scaler values, coefficients, intercepts, and two confidence
thresholds are learned.

Generate out-of-fold predictions using five-fold grouped cross-validation:

- fixed seed: `20261007`;
- grouping key for CocoGlide: COCO source identifier, keeping each real/edited
  pair in the same fold;
- grouping key for DiffusionDB: image identifier;
- class proportions should be kept as even as grouping permits;
- all standardization and model fitting occur inside each training fold.

Select `c1` and `c2` jointly from the fixed grid `0.50, 0.51, ..., 0.95` using
only out-of-fold predictions. Eligible pairs must provide at least 80%
end-to-end coverage. Among eligible pairs, choose by the following fixed order:

1. highest end-to-end macro F1, with inconclusive samples counted as misses;
2. highest balanced accuracy;
3. highest coverage;
4. smallest `c1 + c2`;
5. smallest `c1`, then smallest `c2`.

If no threshold pair reaches 80% coverage, the calibration fails and the
policy must not be evaluated as the primary hierarchy. The protocol may then
be revised under a new version rather than weakening the criterion silently.

After selecting both thresholds, refit the scaler and the two models on the
complete calibration corpus and serialize the frozen policy.

## Policy artifact

The JSON policy must contain at least:

- schema and protocol versions;
- ordered feature names and validation ranges;
- detector names, upstream revisions, checkpoint hashes, and score
  definitions;
- calibration-manifest and calibration-result SHA-256 values;
- sample counts and class counts;
- cross-validation seed, grouping rule, model configuration, and selected
  confidence thresholds;
- feature means and standard deviations;
- both models' coefficients and intercepts;
- software versions used for calibration;
- out-of-fold coverage and metrics;
- creation timestamp and an explicit note that the current three-class
  validation result was not used for fitting.

The runtime classifier must evaluate this artifact without requiring
scikit-learn. Scikit-learn may be an experiment-only dependency used to fit
and verify the serialized coefficients.

## Implementation boundary

The implementation will add a separate entry point, tentatively
`classify_hierarchical(runs, policy)`. It must not alter the output of
`classify(runs)`.

The hierarchical result must retain:

- the final label or inconclusive state;
- an explicit reason identifying the stage that decided or abstained;
- the unchanged detector runs;
- policy version and hash;
- standardized input features;
- `p_artificial`;
- `p_synthetic_given_artificial` when level 2 is reached;
- the two confidence thresholds.

Unit tests must cover policy validation, feature extraction, all three final
classes, both abstention regions, detector errors, missing targets, missing
metadata, non-finite values, coefficient evaluation, and proof that the
original Boolean classifier remains unchanged.

## Retrospective comparison

After the policy artifact is frozen, apply it to the already versioned
768-image result in
[`results/three-class-validation-clean-cache.csv`](results/three-class-validation-clean-cache.csv).
No GPU inference is required because every required feature is present.

This comparison answers whether the new aggregation makes better use of the
same detector outputs. It is not fully blind: the baseline result informed the
choice to investigate a hierarchy and the inclusion of localization extent.
No coefficient, confidence threshold, feature, or class rule may be changed
after inspecting the hierarchical predictions on these 768 rows.

Report the original Boolean and hierarchical policies side by side using:

- the complete true-class by outcome table;
- 3 by 3 confusion matrix for conclusive predictions;
- end-to-end accuracy, balanced accuracy, macro precision, macro recall, and
  macro F1 with abstentions retained as misses;
- conclusive-only accuracy and macro F1 as secondary metrics;
- coverage and abstention rate overall and by true class;
- per-class precision, recall, and F1;
- level-1 real-versus-artificial metrics;
- level-2 synthetic-versus-edited metrics, conditional on true artificial
  samples and separately end to end;
- paired, source-group bootstrap 95% intervals for metric differences from the
  Boolean baseline, using 2,000 resamples and seed `20261007`.

## Prespecified ablations

The primary result always uses the full three-feature policy. Two descriptive
ablations will be fitted on the same calibration corpus and reported without
replacing the primary result:

1. omit `distildire_score`, measuring whether DistilDIRE adds information;
2. omit `dinolizer_marked_area`, measuring the value of localization extent.

These ablations prevent claiming a multi-detector benefit if the fitted policy
effectively relies on DinoLizer alone. They are diagnostic and do not authorize
post-evaluation selection of whichever model performs best.

## Engineering acceptance criteria

The retrospective hierarchy is considered suitable for continued development
only if it meets all of the following on the frozen 768-row comparison:

- zero aggregation errors on structurally valid rows;
- at least 80% coverage;
- at least 70% end-to-end balanced accuracy;
- at least 65% end-to-end macro F1;
- at least 50% recall for every class;
- higher balanced accuracy and macro F1 than the Boolean baseline.

These are engineering gates, not proof of broad generalization. Missing them
is a valid experimental result and must not be repaired by tuning on the
comparison set.

## Independent confirmation

Because the three-class result has already been inspected, an external corpus
selected after the policy is frozen is required before making a general
performance claim. That corpus must:

- contain all three classes and keep source-related images in the same split;
- use real and edited images from sources not used for policy calibration;
- include at least one synthetic generator and one editing method absent from
  calibration;
- preserve a deterministic manifest, source revisions, input hashes, and
  balanced class counts;
- run both adapters again rather than reuse class-dependent cached features;
- apply the frozen policy without refitting.

Dataset selection and sample count will be frozen in a separate confirmation
protocol before downloading model outputs. Until that confirmation is
complete, the hierarchy will be described as an internally validated
experimental component rather than a universal detector.

## Required artifacts

- calibration manifest and complete raw detector result;
- grouped fold assignments and out-of-fold predictions;
- frozen hierarchical-policy JSON and SHA-256;
- calibration metrics and threshold-selection table;
- retrospective predictions, metrics, confusion tables, and figures;
- ablation results clearly labeled as descriptive;
- environment metadata, logs, and artifact checksum list;
- final report documenting successes, failures, and limitations.

## Execution order

1. implement manifest preparation and offline calibration tooling;
2. add tests for deterministic selection, grouping, model fitting, policy
   serialization, and pure-Python coefficient evaluation;
3. review and commit the calibration manifest before inference;
4. run a three-image smoke test in Colab;
5. run the complete 768-image calibration inference;
6. fit and freeze the policy from grouped out-of-fold predictions;
7. implement the runtime hierarchical classifier against the frozen schema;
8. apply the frozen policy to the existing 768-row comparison;
9. report primary, ablation, and paired-bootstrap results;
10. design the independent confirmation protocol.

The commands for the first manifest-freezing phase are documented in
[`hierarchical-aggregation-runbook.md`](hierarchical-aggregation-runbook.md).

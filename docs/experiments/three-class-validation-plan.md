# Three-class pipeline validation plan

This experiment will evaluate the complete DiffDetect decision pipeline with
the two implemented external adapters. DistilDIRE will provide synthetic-image
evidence, DinoLizer will provide edited-image evidence, and the existing
`classify` rule will combine their independent decisions into `real`,
`synthetic`, `edited`, or an inconclusive outcome.

The purpose is not to tune either detector again. It is to measure whether the
frozen components work together as a three-class system, especially when one
detector responds to an image from the other detector's target class.

## Frozen system

No threshold, preprocessing step, model weight, or classification rule may be
changed after the evaluation manifest is frozen.

### Synthetic evidence

- Adapter: `DistilDIREDetector`
- Target: `DetectionTarget.SYNTHETIC`
- Upstream commit: `5de48cf`
- Classifier checkpoint SHA-256:
  `e6b76a13ae49eb83d39fb9b1f7de86bf9f63d1dddf0225ca7ad03690e6bc53bc`
- ADM checkpoint MD5: `fd9dd2335b8736d521de0aed54bd90ca`
- Threshold: `0.9877818822860718`
- ADM inference: `ddim20`, float32

### Edited evidence

- Adapter: `DinoLizerDetector`
- Target: `DetectionTarget.EDITED`
- Upstream commit: `3241ce530a685e6e0560db4e0d8aaa9f28de6fc6`
- Checkpoint: `epoch=36_val_loss=0.2130.ckpt`
- Checkpoint SHA-256:
  `96cb26f2536919d67b75a5aff195b683a1d8590fe596e6f1260ab7950883e2a2`
- Score: 99th percentile of the processed class-2 probability map
- Threshold: `0.5946570634841919`
- Model inference: float16

### Final decision rule

Both adapters must run on every image. Their Boolean decisions map to the final
outcome as follows:

| DistilDIRE | DinoLizer | Final outcome | Reason |
| --- | --- | --- | --- |
| negative | negative | `real` | `no_evidence` |
| positive | negative | `synthetic` | `synthetic_evidence` |
| negative | positive | `edited` | `edited_evidence` |
| positive | positive | inconclusive | `conflicting_evidence` |

Any detector error also produces an inconclusive result with reason
`detector_error`. Inconclusive and failed samples remain in every end-to-end
metric denominator.

## Evaluation corpus

The balanced corpus will contain 768 images: 256 per true class. Input images
will remain outside the Git repository. A manifest with immutable identifiers,
source revisions, labels, dimensions, and file SHA-256 values will be committed
before model inference begins.

### Real class

Use all 256 authentic COCO images from the frozen
[CocoGlide](https://huggingface.co/datasets/nebula/CocoGlide/tree/275f045df7caa2544dd28de7fa86b044ab661bd0)
validation split. These images were not part of DinoLizer calibration.

### Edited class

Use the 256 GLIDE-inpainted counterparts from the same CocoGlide validation
split. Keeping the authentic and edited members together preserves the
source-separated split that was used for DinoLizer validation.

The frozen CocoGlide inputs are identified by:

- dataset revision: `275f045df7caa2544dd28de7fa86b044ab661bd0`;
- split manifest SHA-256:
  `8311d0e96ccb83bfbfdb2ee981d604515c921701c6b4e7210c0646fc6217e21c`;
- prepared sample metadata SHA-256:
  `5bb40eb0cd8b8ad73256bde8d82d5f95513749414da5e9c67a59e23aac76454f`.

### Synthetic class

Select 256 Stable Diffusion images from
[DiffusionDB 2M](https://huggingface.co/datasets/poloclub/diffusiondb/tree/fb620fbe49fa4420e0734bd9c0df11f51176b61f)
at revision `fb620fbe49fa4420e0734bd9c0df11f51176b61f`. The revision is fixed
before selection and must be passed explicitly when data is downloaded.

The source archive is selected without inspecting model outputs:

1. Calculate SHA-256 of the protocol identifier
   `diffdetect-three-class-v1`, obtaining
   `3dd9db8593d505769c30a1a044b3533fbafad42ad9716c1c3a51252d2f38dcf7`.
2. Interpret the first 16 hexadecimal digits as an integer, calculate modulo
   2,000, and add one. This selects DiffusionDB part `001079`.
3. Keep entries whose image and prompt NSFW scores are finite and no greater
   than `0.1`, whose dimensions are positive, and whose image file exists.
4. Rank eligible filenames by SHA-256 of
   `diffdetect-three-class-v1:{image_name}` and take the first 256.
5. If one part does not contain 256 eligible files, continue through subsequent
   part numbers cyclically until the target is reached. This contingency must
   be recorded in the manifest.

The earlier DistilDIRE baseline used 20 DiffusionDB images but did not version
their identifiers. If those identifiers can be recovered from the preserved
notebook or Drive artifacts, their overlap with the new manifest will be
reported. The new selection remains fixed regardless of detector output.

## Execution protocol

### Phase 1: shared-environment smoke test

1. Start a clean GPU runtime and record Python, package, CUDA, driver, and GPU
   versions.
2. Verify every external repository revision and checkpoint digest before
   loading a model.
3. Construct both official adapters in one Python process and one
   `DetectorRunner` in the order DistilDIRE, then DinoLizer.
4. Run one real, one synthetic, and one edited image through both adapters and
   `classify`.
5. Confirm that lazy imports, model reuse, localization-map dimensions, and
   detector metadata are valid.
6. Record initialization time and GPU allocation separately from warm
   per-image inference.

The smoke images are part of the frozen evaluation corpus and are not removed
from the final run. If the shared environment fails, diagnose and document the
technical incompatibility before changing any dependency or adapter code.

### Phase 2: frozen evaluation

1. Process the manifest in its committed order.
2. Run both detectors on every image, even after one detector returns a
   positive result.
3. Save each raw detector score, threshold, Boolean decision, duration, error,
   and DinoLizer metadata before applying the final rule.
4. Save the final label and `DecisionReason` returned by `classify`.
5. Do not retry, exclude, or relabel failed and conflicting samples silently.
   Any justified retry must be recorded as a separate attempt.
6. Preserve complete artifacts in Drive and commit only compact manifests,
   derived CSV/JSON results, hashes, and documentation.

Probability maps do not need to be committed. Representative error-analysis
maps may be preserved outside Git with their hashes and sample identifiers.

## Required output fields

The per-image result must include at least:

- protocol version and sample identifier;
- source dataset, source revision, true class, and input SHA-256;
- DistilDIRE score, threshold, decision, duration, and error;
- DinoLizer score, threshold, decision, duration, error, marked area,
  processed size, and window count;
- final label, decision reason, inconclusive flag, and total duration;
- attempt number and software-environment identifier.

## Metrics

Report results on all 768 samples and separately by true class.

### End-to-end classification

- a true-class by outcome table with columns `real`, `synthetic`, `edited`,
  `conflicting_evidence`, and `detector_error`;
- the ordinary 3 by 3 confusion matrix for conclusive predictions, clearly
  labeled as conditional on coverage;
- overall accuracy with every inconclusive result counted as incorrect;
- balanced accuracy, macro precision, macro recall, and macro F1 with
  inconclusive results retained as misses for their true classes;
- conclusive-only accuracy and macro F1, reported only as secondary metrics;
- coverage: conclusive samples divided by all samples;
- conflict rate and detector-error rate, overall and by true class.

### Detector behavior

- DistilDIRE sensitivity on synthetic images and positive rates on real and
  edited images;
- DinoLizer sensitivity on edited images and positive rates on real and
  synthetic images;
- the frequency of each joint decision pattern: negative/negative,
  positive/negative, negative/positive, and positive/positive;
- score distributions by true class without comparing raw scores across the
  two detectors.

### Runtime

- model initialization time;
- per-detector mean, median, and 95th-percentile inference time;
- complete pipeline mean, median, and 95th-percentile time;
- peak allocated and reserved GPU memory after both models are loaded.

## Validity rules

The experiment is technically valid only if the frozen manifest contains 256
verified inputs per class, all model and data revisions are recorded, and no
threshold or decision rule is changed after inference starts. Detector errors
and conflicts are findings rather than grounds for excluding samples.

No minimum accuracy is declared in advance. The experiment measures the
current system and determines the next engineering action:

- acceptable coverage and class behavior support moving to a user-facing CLI;
- high cross-target conflict motivates revising the evidence-combination rule
  or evaluating an additional detector;
- detector errors or dependency conflicts motivate an integration fix before
  any quality claim.

## Known limitations

- The three classes come from different generator and dataset families.
- Real and edited images are paired COCO content, while synthetic images are
  unpaired Stable Diffusion outputs.
- File format, resolution, and content differences may act as shortcuts.
- The component thresholds were calibrated on different datasets.
- Reusing previously validated component datasets makes this a frozen system
  evaluation, not an entirely new blind benchmark.
- The evaluated generator families do not represent all diffusion models or
  editing methods.

These limitations must accompany the final metrics and prevent interpreting
the result as universal real-versus-synthetic-versus-edited performance.

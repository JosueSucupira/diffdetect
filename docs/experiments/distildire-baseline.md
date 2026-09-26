# DistilDIRE baseline

This experiment checked the ImageNet DistilDIRE checkpoint before its adapter
was added to DiffDetect. It is a preliminary baseline, not a final evaluation.

## Reproducible setup

- Google Colab with a Tesla T4 GPU
- Python 3.11.13
- PyTorch 2.3.1+cu121
- torchvision 0.18.1+cu121
- DistilDIRE commit `5de48cf`
- DistilDIRE checkpoint SHA-256
  `e6b76a13ae49eb83d39fb9b1f7de86bf9f63d1dddf0225ca7ad03690e6bc53bc`
- ADM checkpoint MD5 `fd9dd2335b8736d521de0aed54bd90ca`
- ADM inference with `ddim20`, in float32
- Official preprocessing: RGB, scale to `[-1, 1]`, resize to 256, and center
  crop to 256 by 256

## Threshold calibration

The official threshold of 0.5 produced 51 false positives in 100 real
Imagenette images. The sigmoid scores still separated the evaluated real and
synthetic samples well: the calibration ROC-AUC was 0.9755.

Youden's J statistic selected an experimental threshold of
`0.9877818822860718` from 100 real Imagenette images and 20 ADM images. On that
same calibration set, the threshold produced 93% specificity and 100%
sensitivity. These values are optimistic because the threshold was selected on
the evaluated data.

## Separate validation

The fixed threshold was then evaluated without recalibration.

| Synthetic source | Real images | Synthetic images | Specificity | Sensitivity | Balanced accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADM, new seed | 100 | 20 | 0.90 | 0.95 | 0.925 | not recorded |
| Stable Diffusion, DiffusionDB | 100 | 20 | 0.90 | 1.00 | 0.950 | 0.9915 |

The 100 real validation images were separate from the real calibration images.
The DiffusionDB images were also independent from the ADM images used for
calibration.

## Framework integration check

The adapter from commit `2c8f254` was executed in the same Colab environment
against two DiffusionDB images that had already been evaluated by the manual
pipeline.

| Run | Score | Synthetic decision | Duration |
| --- | ---: | --- | ---: |
| First, including model loading | 0.994035 | yes | 14.66 s |
| Second, reusing loaded models | 0.996187 | yes | 0.6651 s |

The earlier manually reported scores were 0.9940 and 0.9962 after rounding.
The adapter therefore reproduced the manual pipeline and kept its loaded models
between predictions. GPU allocation remained at 4.32 GB during the second run
because the notebook still held the manually loaded models alongside the
adapter's copies. This value does not represent a clean adapter-only session.

## Limits

- The samples are too small to establish a universal operating threshold.
- Synthetic validation covers ADM and one Stable Diffusion dataset only.
- Ten percent of the separate real images remained false positives.
- The calibrated score is evidence for the synthetic class, not proof that an
  image is synthetic and not a probability that can be compared directly with
  another detector's score.

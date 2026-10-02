# DinoLizer baseline

This experiment checks whether DinoLizer can provide localization evidence for
the edited-image class in DiffDetect. The technical smoke test, an initial
paired behavior check, and image-level calibration have been completed. The
calibrated decision rule is frozen; separate validation remains pending.

## Reproducible setup

- Google Colab with a Tesla T4 GPU and 15,360 MiB of VRAM
- Python 3.13.15
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- CUDA 12.8 and NVIDIA driver 580.82.07
- timm 1.0.29
- DinoLizer commit `3241ce530a685e6e0560db4e0d8aaa9f28de6fc6`
- Official `BFREE_dino2reg4` checkpoint:
  `epoch=36_val_loss=0.2130.ckpt`
- Checkpoint size: 386,800,693 bytes
- Checkpoint SHA-256:
  `96cb26f2536919d67b75a5aff195b683a1d8590fe596e6f1260ab7950883e2a2`
- DINOv2 backbone: `vit_base_patch14_reg4_dinov2.lvd142m`
- Official `Wrapper5crops` inference at an input size of 504 pixels
- Model parameters loaded in float16 on the GPU

The checkpoint was first inspected on the CPU with safe weight-only loading.
It contained 235 state-dictionary entries and approximately 350.72 MB of
tensor data. The weights were then loaded through DinoLizer's custom
`Wrapper5crops.load_state_dict` method, matching the official inference code.

## Technical smoke test

The unchanged official `test.py` script completed successfully for all 16
example images distributed with the project. Every resulting mask:

- had the same spatial dimensions as its input image;
- contained only the binary values 0 and 255;
- contained a non-empty predicted edited region.

The model-loading step took 0.51 seconds after the DINOv2 backbone was cached
and allocated approximately 0.18 GB of GPU memory. The complete official
subprocess took 26.36 seconds for 16 example images, including initialization;
the inference progress itself took approximately 14 seconds.

The warning about torchvision's deprecated `ToTensor` transform did not affect
execution. It records an upstream compatibility issue to address in a future
adapter without changing the experiment's official preprocessing.

## Paired behavior check

The filenames of the official examples include their source COCO image IDs.
The corresponding 16 original COCO `train2017` photographs were retrieved
from the official COCO S3 endpoint. The unchanged inference script was then
run on those photographs, allowing each original to be compared with its
FluxKontext- or Qwen-edited version.

The exploratory image-level score in this check is the proportion of pixels
marked as edited by the official binary mask. It is a localization summary,
not a calibrated probability.

| COCO ID | Original area | Edited area | Difference |
| --- | ---: | ---: | ---: |
| 000000113233 | 0.0000% | 9.1759% | +9.1759 pp |
| 000000167577 | 0.0000% | 30.3305% | +30.3305 pp |
| 000000368603 | 0.5906% | 26.6138% | +26.0232 pp |
| 000000374858 | 0.0000% | 25.5508% | +25.5508 pp |
| 000000391596 | 0.0000% | 2.8542% | +2.8542 pp |
| 000000393730 | 0.0000% | 17.3504% | +17.3504 pp |
| 000000414659 | 0.0000% | 1.4471% | +1.4471 pp |
| 000000419391 | 0.0000% | 20.5052% | +20.5052 pp |
| 000000424837 | 0.0000% | 28.1649% | +28.1649 pp |
| 000000460755 | 0.3600% | 13.4370% | +13.0770 pp |
| 000000501594 | 0.0000% | 18.5690% | +18.5690 pp |
| 000000503097 | 0.0000% | 9.7432% | +9.7432 pp |
| 000000503170 | 0.0000% | 3.5304% | +3.5304 pp |
| 000000523641 | 0.0000% | 22.8326% | +22.8326 pp |
| 000000563809 | 0.0765% | 2.4422% | +2.3656 pp |
| 000000577975 | 0.0000% | 20.0261% | +20.0261 pp |

| Distribution | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Original photographs | 0.0642% | 0.0000% | 0.0000% | 0.5906% |
| Edited images | 15.7858% | 17.9597% | 1.4471% | 30.3305% |

The edited area was greater for the edited member of all 16 pairs. There were
no ties. Treating each image independently, the exploratory ROC-AUC of the
marked-area score was 1.0 on these 32 images. The original-image subprocess
took 13.29 seconds in total, with the inference progress itself taking about
3.8 seconds.

## Interpretation and limits

The behavior check confirms that the official model, checkpoint, and
preprocessing execute correctly in the tested Colab environment. It also
shows that the marked-area proportion is a plausible candidate for an
image-level edited score.

These results do not establish a decision threshold or an expected production
accuracy:

- the edited images are examples supplied by the DinoLizer project;
- only 16 source pairs were evaluated;
- the edited files are PNG images while the originals are JPEG images;
- several edited images have different dimensions from their originals;
- no ground-truth manipulation masks were available for pixel-level IoU or F1;
- the score has not been calibrated and evaluated on separate splits.

These limitations motivated the separate CocoGlide protocol described below.
Calibration images and validation images, including derivatives of the same
source photograph, remain in different splits. Adapter implementation remains
pending until the frozen decision rule is checked on the validation split.

## CocoGlide calibration protocol

Calibration uses the reformatted `nebula/CocoGlide` dataset at revision
`275f045df7caa2544dd28de7fa86b044ab661bd0`. It contains 512 source pairs. Each
pair has one authentic COCO image and one GLIDE-inpainted image with a
localization mask.

The source IDs were sorted by the SHA-256 digest of
`diffdetect-cocoglide-v1:{coco_id}`. The first 256 pairs were assigned to
calibration and the remaining 256 to validation. This procedure keeps both
members of a source pair in the same split and produces no source overlap.

- Split manifest SHA-256:
  `8311d0e96ccb83bfbfdb2ee981d604515c921701c6b4e7210c0646fc6217e21c`
- Prepared sample metadata SHA-256:
  `5bb40eb0cd8b8ad73256bde8d82d5f95513749414da5e9c67a59e23aac76454f`
- Calibration result SHA-256:
  `6ec4fa85098466cdafcb592afdd1be59c9c474ac09679f9da557522986e86244`

The calibration run used a later Colab image than the smoke test:

- Python 3.13.15
- PyTorch 2.11.0+cu130
- torchvision 0.26.0+cu130
- CUDA 13.0
- NumPy 2.1.3
- timm 1.0.29
- Tesla T4

The inference implementation reproduced the previously recorded official mask
area of 9.1759% as 9.1797%, a difference of 0.0038 percentage point. For
256-pixel CocoGlide inputs, it follows the official resize to 1016 pixels and
uses 25 overlapping 504-pixel windows with a stride of 128. Probability maps
and binary masks are returned to the original dimensions for localization
metrics.

## Image-level score selection

Six candidate summaries were compared on the 512 calibration images. For each
candidate, the table reports its ROC-AUC and the result at the threshold that
maximized Youden's J statistic on calibration.

| Candidate score | ROC-AUC | Threshold | Sensitivity | Specificity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Marked area, processed size | 0.9845 | 0.003767 | 0.9727 | 0.9258 | 0.9492 |
| Marked area, original size | 0.9846 | 0.003754 | 0.9727 | 0.9258 | 0.9492 |
| Mean inpainted probability | 0.9727 | 0.031757 | 0.9219 | 0.9336 | 0.9277 |
| 95th probability percentile | 0.9637 | 0.264661 | 0.8398 | 0.9648 | 0.9023 |
| 99th probability percentile | **0.9891** | **0.5946570634841919** | 0.9453 | 0.9570 | 0.9512 |
| Maximum probability | 0.9889 | 0.753046 | 0.9609 | 0.9492 | 0.9551 |

The 99th percentile was selected because it had the highest calibration
ROC-AUC and is less sensitive to a single extreme pixel than the maximum. The
image-level edited score is therefore the 99th percentile of the DinoLizer
class-2 probability map. An image supplies edited evidence when that score is
greater than or equal to `0.5946570634841919`.

This score is a detector-specific summary, not the probability that an entire
image is edited. The score definition and threshold are frozen before the
validation split is evaluated.

At the frozen calibration operating point, the confusion matrix contained 242
true positives, 245 true negatives, 11 false positives, and 14 false
negatives. Precision was 0.9565 and accuracy was 0.9512.

## Calibration localization and runtime

Localization metrics were calculated on the 256 manipulated calibration
images after returning predicted masks to their original 256-pixel size.

| Metric | Macro mean | Median | Micro |
| --- | ---: | ---: | ---: |
| IoU | 0.7441 | 0.8384 | 0.7851 |
| F1 | 0.8246 | 0.9121 | 0.8796 |

Two manipulated images produced empty predicted masks. The correlation between
ground-truth and predicted edited-area proportions was 0.9202.

All 512 calibration images completed without errors in 8.54 minutes. Mean
model inference time was 0.918 seconds per image, the median was 0.894 seconds,
and the 95th percentile was 1.013 seconds. The raw derived results are stored
in [`results/dinolizer-cocoglide-calibration.csv`](results/dinolizer-cocoglide-calibration.csv).

## Remaining validation

The frozen 99th-percentile score and threshold must now be applied unchanged
to the 256 authentic and 256 manipulated images in the held-out validation
split. Validation must report the image-level confusion matrix, sensitivity,
specificity, precision, accuracy, balanced accuracy, and ROC-AUC, as well as
pixel-level IoU and F1 for the manipulated images. No validation result may be
used to revise the score or threshold.

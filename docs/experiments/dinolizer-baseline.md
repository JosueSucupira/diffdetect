# DinoLizer baseline

This experiment checks whether DinoLizer can provide localization evidence for
the edited-image class in DiffDetect. The technical smoke test and an initial
paired behavior check have been completed. Image-level calibration and
independent validation remain pending.

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

The next experiment must use traceable data that are separate from these
examples. Calibration images and validation images, including derivatives of
the same source photograph, must remain in different splits. Only after that
test can DiffDetect define an image-level score, freeze a threshold, and
implement a DinoLizer adapter targeting `DetectionTarget.EDITED`.

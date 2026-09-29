# DinoLizer baseline plan

This experiment will determine whether DinoLizer can supply edited-image
evidence to DiffDetect. It must be completed before an adapter is implemented.

## Why DinoLizer is the first candidate

DinoLizer has a public inference implementation, pretrained weights, and an
individual-image script. It targets generative inpainting localization, which
fits the edited class used by DiffDetect.

The other candidates are less suitable for the first integration:

- The repository announced by the X-Edit paper is not currently available at
  its published URL.
- DiQuID, now published as SAGI, provides an inpainting generation pipeline
  and dataset rather than a detector ready for single-image inference.

Official DinoLizer repository:
<https://github.com/anonyme610/dinolizer>

## Important output mismatch

DiffDetect expects an edited detector to return:

- a Boolean detection decision;
- an optional image-level score and threshold;
- an optional localization map.

The official DinoLizer inference script instead produces pixel-level logits.
It applies a softmax and marks a pixel as manipulated when the probability of
class 2 is greater than 0.5. The script saves the resulting binary mask but
does not define an image-level score or a rule for deciding whether the whole
image is edited.

The experiment must preserve the probability map and evaluate candidate
image-level summaries, such as the proportion of pixels marked as edited. No
image-level threshold will be added to the framework before it is calibrated
and checked on separate data.

## Phase 1: technical smoke test

Use a fresh Google Colab runtime with a Tesla T4 GPU.

1. Record the Python, PyTorch, torchvision, CUDA, and GPU versions.
2. Clone DinoLizer and record the exact commit.
3. Install only the dependencies required for inference.
4. Download the official pretrained checkpoint and record its size and hash.
5. Run the official inference script on its supplied examples.
6. Confirm that every output mask has the expected dimensions and finite
   values.
7. Measure model-loading time, warm inference time, and GPU memory usage.

The official environment pins Python 3.9 and CUDA 11.8. Compatibility with the
current Colab runtime must be tested rather than assumed.

## Phase 2: behavior check

Run the unchanged inference pipeline on a small, traceable set containing:

- original photographs;
- paired inpainted versions of those photographs;
- the corresponding ground-truth masks when available.

For every image, retain:

- the probability map before binarization;
- the binary localization mask;
- the predicted edited-area proportion;
- inference duration;
- source and expected class.

This phase checks whether original and inpainted images show meaningfully
different output distributions. It does not establish a final threshold.

## Phase 3: preliminary calibration and separate validation

If the behavior check succeeds:

1. Select an image-level score using only a calibration split.
2. Select its decision threshold using only that split.
3. Freeze both choices.
4. Evaluate them on separate original and inpainted images.
5. Report image-level sensitivity, specificity, balanced accuracy, and
   ROC-AUC.
6. Report pixel-level F1 and IoU when ground-truth masks are available.

Calibration images, validation images, and derived versions of the same source
image must not cross splits.

## Adapter acceptance criteria

The DinoLizer adapter can be implemented only after the experiment identifies:

- a reproducible environment and pinned upstream commit;
- an official checkpoint with a recorded hash;
- the exact preprocessing and localization output;
- a defensible image-level score and threshold;
- behavior on validation images not used for calibration;
- expected loading time, inference time, and GPU memory use.

If accepted, the adapter will target `DetectionTarget.EDITED`, return the
localization mask through `DetectionResult.localization_map`, and keep the
external DinoLizer code and weights outside the DiffDetect repository.

# diffdetect
A modular framework for classifying images as real, diffusion-generated, or diffusion-edited.

## Development status

Steps 1-4 define the common adapter interface, a registry that constructs
adapters by name, a runner that executes them on an image, and a three-class
decision rule. Step 5 adds an adapter for the external DistilDIRE ImageNet
detector. Step 6 adds an experimental adapter for the external DinoLizer
localization model.

The frozen three-class baseline has also been evaluated on 768 images. Both
adapters completed without technical errors, but the conservative Boolean
aggregation achieved 6.64% end-to-end accuracy with 40.62% coverage and a
59.38% conflict rate. This negative result and the next hierarchical
aggregation step are documented in
[`docs/experiments/three-class-validation.md`](docs/experiments/three-class-validation.md).
The follow-up protocol is frozen in
[`docs/experiments/hierarchical-aggregation-plan.md`](docs/experiments/hierarchical-aggregation-plan.md)
before any aggregation parameters are fitted.

An adapter inherits from `BaseDetector`, declares a name and whether it looks
for fully synthetic or edited images, and implements `predict(image)`. The input
is an RGB [Pillow](https://pillow.readthedocs.io/) image. The result always
contains a Boolean detection decision. It may also contain the detector's raw
score, the threshold used for that decision, a localization map, and
detector-specific metadata.

```python
from PIL import Image
from diffdetect import BaseDetector, DetectionResult, DetectionTarget


class MySyntheticDetector(BaseDetector):
    name = "my_synthetic_detector"
    target = DetectionTarget.SYNTHETIC

    def __init__(self, model):
        self.model = model

    def predict(self, image: Image.Image) -> DetectionResult:
        score = float(self.model.score(image))
        threshold = 0.5
        return DetectionResult(
            detected=score >= threshold,
            score=score,
            threshold=threshold,
        )
```

The example shows how a model will be wrapped; `MySyntheticDetector` is not an
implemented detector. The `real` class will be decided by the framework after
both synthetic and edited evidence have been evaluated. Scores remain on each
detector's own scale and cannot be compared directly across models.

Registering a detector makes its class available by name. Model construction
occurs later, when `create` is called, so each adapter can receive its own model
or configuration parameters:

```python
from diffdetect import DetectorRegistry

registry = DetectorRegistry()
registry.register(MySyntheticDetector)
print(registry.names())  # ('my_synthetic_detector',)
detector = registry.create("my_synthetic_detector", model=my_model)
```

`my_model` in this example stands for a model supplied by the user. Separate
registry instances maintain separate lists of adapters.

Use `DetectorRunner` to execute one or more constructed detectors. It accepts a
file path or a Pillow image, converts it to RGB, and gives each detector its own
copy. Every run records the detector name, target, result or error, and duration:

```python
from diffdetect import DetectorRunner

runner = DetectorRunner([detector])
runs = runner.run("image.png", selected=["my_synthetic_detector"])
for run in runs:
    print(run.name, run.target, run.status, run.result, run.error)
```

When `selected` is omitted, all loaded detectors run in their construction
order. A detector failure is recorded and does not prevent the others from
running. An unreadable input image raises an error before execution.

The `classify` function combines the individual runs into `real`, `synthetic`,
or `edited` when both targets were evaluated successfully:

```python
from diffdetect import classify

decision = classify(runs)
print(decision.label, decision.reason)
```

With only the synthetic detector from the example above, this decision is
inconclusive because the edited target has not been checked.

Within each target, any positive detector supplies evidence for that target.
Evidence for both `synthetic` and `edited` is a conflict. A detector error or
missing target also prevents a three-class decision. In those cases,
`decision.label` is `None` and `decision.reason` explains why; `None` is an
inconclusive status, not a fourth class. A `real` label means that the included
detectors found no evidence for either target, not that authenticity was proven.
The decision rule does not compare raw scores from different detectors.

## DistilDIRE adapter

`DistilDIREDetector` wraps the official ImageNet DistilDIRE implementation as
synthetic-image evidence. DiffDetect does not redistribute its source code or
model weights. Clone the
[official repository](https://github.com/miraflow/DistilDIRE), download its
ImageNet classifier and ADM checkpoints, and follow the external project's
CC BY-NC 4.0 license.

The validated Colab environment used Python 3.11, PyTorch 2.3.1 with CUDA 12.1,
and torchvision 0.18.1. `blobfile` and `mpi4py` are also required by the
external implementation. The adapter imports these dependencies only when its
first prediction loads the models, so the rest of DiffDetect remains usable
without PyTorch.

```python
from diffdetect import DetectorRunner, DistilDIREDetector

detector = DistilDIREDetector(
    repository_path="/content/DistilDIRE",
    classifier_weights=(
        "/content/DistilDIRE/models/imagenet-distil-dire-11e.pth"
    ),
    adm_weights="/content/DistilDIRE/models/256x256-adm.pt",
    threshold=0.9877818822860718,
    device="cuda",
)

runner = DetectorRunner([detector])
run = runner.run("image.jpg")[0]
print(run.result.score, run.result.detected)
```

The threshold in this example came from a small preliminary calibration and is
not a universal default. The adapter therefore requires the threshold to be
provided explicitly. The complete protocol, metrics, hashes, and limitations
are recorded in
[`docs/experiments/distildire-baseline.md`](docs/experiments/distildire-baseline.md).

## DinoLizer adapter

`DinoLizerDetector` wraps the official DinoLizer localization model as
edited-image evidence. DiffDetect does not redistribute its source code or
weights. The validated integration uses
[DinoLizer commit `3241ce5`](https://github.com/anonyme610/dinolizer/tree/3241ce530a685e6e0560db4e0d8aaa9f28de6fc6)
and the official `BFREE_dino2reg4` checkpoint named
`epoch=36_val_loss=0.2130.ckpt`.

Clone and pin the external repository:

```bash
git clone https://github.com/anonyme610/dinolizer.git ../DinoLizer
git -C ../DinoLizer checkout --detach 3241ce530a685e6e0560db4e0d8aaa9f28de6fc6
```

Download the checkpoint from the
[official model link](https://nextcloud.univ-lille.fr/index.php/s/HS6CLT73DZpZEyq),
keep it outside this repository, and verify the exact validated file before
running it:

```bash
sha256sum ../models/dinolizer/epoch=36_val_loss=0.2130.ckpt
# 96cb26f2536919d67b75a5aff195b683a1d8590fe596e6f1260ab7950883e2a2
```

Install PyTorch and torchvision for the CUDA version available on the target
machine, then install the remaining modules imported by the pinned upstream
code:

```bash
python -m pip install numpy timm PyYAML albumentations einops safetensors
```

The integration was validated on a Tesla T4 with Python 3.13.15, timm 1.0.29,
and two Colab runtime images using PyTorch 2.11.0 with torchvision 0.26.0. The
adapter loads these optional dependencies, the external source, and the
checkpoint only on its first prediction. The pretrained DINOv2 backbone used
by `timm` is also downloaded and cached on first use unless it is already
present.

```python
from diffdetect import DetectorRunner, DinoLizerDetector

detector = DinoLizerDetector(
    repository_path="../DinoLizer",
    checkpoint_path=(
        "../models/dinolizer/epoch=36_val_loss=0.2130.ckpt"
    ),
    threshold=0.5946570634841919,
    device="cuda",
)

runner = DetectorRunner([detector])
run = runner.run("image.jpg")[0]
result = run.result

if result is None:
    raise RuntimeError(run.error)

print(result.score, result.threshold, result.detected)
print(result.localization_map.size, result.metadata)
```

The score is the 99th percentile of the class-2 probability map at DinoLizer's
processed resolution. The default threshold is `0.5946570634841919`, selected
on the CocoGlide calibration split; omit the argument to use that default or
pass another value to apply a separately calibrated operating point.
`localization_map` is a floating-point Pillow image in mode `F`, returned at
the original image size with values in the interval `[0, 1]`. Metadata records
the marked-area proportion, original and processed sizes, window count,
504-pixel window size, 128-pixel stride, and the 0.5 pixel threshold.

The default operating point performed well on the source-separated CocoGlide
validation split, but it is not a universal production threshold. Recalibrate
it when the target distribution differs substantially in editing method,
resolution, compression, or manipulated-area size. The complete protocol,
metrics, hashes, and limitations are recorded in
[`docs/experiments/dinolizer-baseline.md`](docs/experiments/dinolizer-baseline.md).

The pinned upstream source files state that they are for nonprofit use and
refer users to the
[GRIP license terms](https://www.grip.unina.it/download/LICENSE_OPEN.txt).
Review those terms before downloading or using the external code and weights.

Install the package locally with `python -m pip install -e .`. Run the contract
tests with `PYTHONPATH=src python -m unittest discover -s tests -v`.

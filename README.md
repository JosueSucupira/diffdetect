# diffdetect
A modular framework for classifying images as real, diffusion-generated, or diffusion-edited.

## Development status

Steps 1-4 define the common adapter interface, a registry that constructs
adapters by name, a runner that executes them on an image, and a three-class
decision rule. Step 5 adds an adapter for the external DistilDIRE ImageNet
detector.

An adapter inherits from `BaseDetector`, declares a name and whether it looks
for fully synthetic or edited images, and implements `predict(image)`. The input
is an RGB [Pillow](https://pillow.readthedocs.io/) image. The result always
contains a Boolean detection decision. It may also contain the detector's raw
score, the threshold used for that decision, and a localization map.

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

Install the package locally with `python -m pip install -e .`. Run the contract
tests with `PYTHONPATH=src python -m unittest discover -s tests -v`.

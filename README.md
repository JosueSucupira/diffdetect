# diffdetect
A modular framework for classifying images as real, diffusion-generated, or diffusion-edited.

## Development status

Steps 1 and 2 define the common adapter interface and a registry that selects
adapters by name. The repository does not yet include a pretrained detector or
image classification pipeline.

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

Install the package locally with `python -m pip install -e .`. Run the contract
and registry tests with `PYTHONPATH=src python -m unittest discover -s tests -v`.

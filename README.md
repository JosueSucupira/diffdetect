# diffdetect
A modular framework for classifying images as real, diffusion-generated, or diffusion-edited.

## Development status

Step 1 defines the common interface for detector adapters. The repository does
not yet include a pretrained detector or image classification pipeline.

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

Install the package locally with `python -m pip install -e .`. Run the contract
tests with `PYTHONPATH=src python -m unittest discover -s tests -v`.

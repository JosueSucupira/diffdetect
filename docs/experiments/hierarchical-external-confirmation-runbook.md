# External confirmation runbook

This runbook starts only after the source datasets and their licenses have been
reviewed. Do not run DistilDIRE or DinoLizer before the candidate catalog and
manifest have been frozen.

## 1. Fill the candidate catalog

Copy
[`results/hierarchical-external-candidate-template.csv`](results/hierarchical-external-candidate-template.csv)
outside the repository, add exactly 384 rows, and preserve the header order.
Use 128 rows per class and follow the diversity quotas in
[`hierarchical-external-confirmation-plan.md`](hierarchical-external-confirmation-plan.md).

Each `source_root` is a short logical name. `relative_path` must be a normalized
path below that root. Record the source metadata hash, image hash, dimensions,
and method provenance before inference.

## 2. Freeze and verify the manifest

Supply every logical root as `NAME=/absolute/path`. The following command is a
template; replace the root names and paths with the selected sources.

```bash
PYTHONPATH=src python experiments/hierarchical_external_validation.py \
  prepare-manifest \
  --catalog /path/to/external-candidate-catalog.csv \
  --root source-a=/path/to/source-a \
  --root source-b=/path/to/source-b \
  --exclude-manifest \
    docs/experiments/results/three-class-validation-manifest.csv \
  --exclude-manifest \
    docs/experiments/results/hierarchical-calibration-manifest.csv \
  --output \
    docs/experiments/results/hierarchical-external-manifest.csv
```

The command verifies every image, all hashes and dimensions, class balance,
source/method quotas, and overlap with both earlier experiments. Review and
commit the resulting manifest and its SHA-256 before opening a GPU runtime.

## 3. Run raw inference

Use the same pinned detector repositories and weights recorded by the
hierarchical calibration experiment. A smoke run selects the earliest frozen
row from each class:

```bash
PYTHONPATH=src python experiments/hierarchical_external_validation.py run \
  --manifest docs/experiments/results/hierarchical-external-manifest.csv \
  --root source-a=/path/to/source-a \
  --root source-b=/path/to/source-b \
  --distildire-repository /path/to/DistilDIRE \
  --distildire-classifier /path/to/imagenet_classificator.pth \
  --distildire-adm /path/to/256x256_diffusion_uncond.pt \
  --dinolizer-repository /path/to/dinolizer \
  --dinolizer-checkpoint /path/to/dinolizer.pth \
  --device cuda \
  --environment-id colab-t4-external-v1 \
  --output /path/to/hierarchical-external-smoke.csv \
  --smoke
```

After validating the three rows, repeat without `--smoke` and with a new output
path. `--resume` may continue an interrupted result that belongs to the same
manifest. The runner never applies the aggregation policy.

## 4. Apply the frozen policy

Run this step only after the complete raw result and metadata hashes have been
verified:

```bash
PYTHONPATH=src python experiments/hierarchical_external_validation.py evaluate \
  --policy docs/experiments/results/hierarchical-policy.json \
  --policy-sha256 \
    c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187 \
  --manifest docs/experiments/results/hierarchical-external-manifest.csv \
  --results /path/to/hierarchical-external.csv \
  --results-metadata /path/to/hierarchical-external.csv.metadata.json \
  --output-dir docs/experiments/results
```

The evaluator writes predictions, metrics, a confusion table and figure, and
an artifact checksum list. It verifies the frozen policy hash and has no model
fitting or threshold-selection command.

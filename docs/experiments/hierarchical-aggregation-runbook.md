# Hierarchical aggregation calibration runbook

This runbook creates the frozen 768-image calibration manifest defined in
[`hierarchical-aggregation-plan.md`](hierarchical-aggregation-plan.md). It does
not run either detector or fit the aggregation policy.

Stop after generating the manifest. Review and commit it before any model
inference begins.

## 1. Start a clean Colab runtime

Mount Drive and clone the experiment branch:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
git clone --branch experiment/hierarchical-aggregation \
  https://github.com/JosueSucupira/diffdetect.git /content/diffdetect
cd /content/diffdetect
python -m pip install -e .
python -m pip install pandas pyarrow
```

The manifest builder reads images and metadata only. A GPU runtime is not
required for this phase.

## 2. Locate the CocoGlide calibration images

The root must directly contain files such as
`000000000285_real.png` and `000000000285_manipulated.png` from the preserved
CocoGlide calibration split:

```bash
export COCOGLIDE_CALIBRATION_ROOT=/content/drive/MyDrive/DiffDetect/datasets/CocoGlide/prepared-v1/calibration/images

test -d "$COCOGLIDE_CALIBRATION_ROOT"
test -f "$COCOGLIDE_CALIBRATION_ROOT/000000000285_real.png"
test -f "$COCOGLIDE_CALIBRATION_ROOT/000000000285_manipulated.png"
```

The committed calibration CSV is verified against SHA-256
`6ec4fa85098466cdafcb592afdd1be59c9c474ac09679f9da557522986e86244`
before its paths are accepted.

## 3. Download the frozen DiffusionDB source

The protocol deterministically selected archive part `001800`, which differs
from validation part `001079`:

```bash
export DIFFUSIONDB_REVISION=fb620fbe49fa4420e0734bd9c0df11f51176b61f
export HIERARCHICAL_DIFFUSIONDB_ROOT=/content/drive/MyDrive/DiffDetect/datasets/DiffusionDB/hierarchical-calibration-v1

mkdir -p "$HIERARCHICAL_DIFFUSIONDB_ROOT"

if [ ! -f "$HIERARCHICAL_DIFFUSIONDB_ROOT/metadata.parquet" ]; then
  wget -O "$HIERARCHICAL_DIFFUSIONDB_ROOT/metadata.parquet.part" \
    "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/$DIFFUSIONDB_REVISION/metadata.parquet?download=true"
  mv "$HIERARCHICAL_DIFFUSIONDB_ROOT/metadata.parquet.part" \
    "$HIERARCHICAL_DIFFUSIONDB_ROOT/metadata.parquet"
fi

if [ ! -f "$HIERARCHICAL_DIFFUSIONDB_ROOT/part-001800.zip" ]; then
  wget -O "$HIERARCHICAL_DIFFUSIONDB_ROOT/part-001800.zip.part" \
    "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/$DIFFUSIONDB_REVISION/images/part-001800.zip?download=true"
  mv "$HIERARCHICAL_DIFFUSIONDB_ROOT/part-001800.zip.part" \
    "$HIERARCHICAL_DIFFUSIONDB_ROOT/part-001800.zip"
fi

unzip -qn "$HIERARCHICAL_DIFFUSIONDB_ROOT/part-001800.zip" \
  -d "$HIERARCHICAL_DIFFUSIONDB_ROOT"
```

Interrupted downloads retain only a `.part` file and are never interpreted as
complete artifacts.

## 4. Generate and verify the manifest

The previous three-class validation manifest is mandatory input. The builder
rejects overlap by sample identifier, source identifier, and image SHA-256.
It also verifies that exclusion manifest against SHA-256
`9dcda4b93d8c73df7b1240629bec209566cd6119bb7f800c2906b8c3826b8777`.

```bash
cd /content/diffdetect

python experiments/hierarchical_aggregation.py prepare-manifest \
  --cocoglide-results docs/experiments/results/dinolizer-cocoglide-calibration.csv \
  --cocoglide-root "$COCOGLIDE_CALIBRATION_ROOT" \
  --diffusiondb-metadata "$HIERARCHICAL_DIFFUSIONDB_ROOT/metadata.parquet" \
  --diffusiondb-root "$HIERARCHICAL_DIFFUSIONDB_ROOT" \
  --exclude-manifest docs/experiments/results/three-class-validation-manifest.csv \
  --output docs/experiments/results/hierarchical-calibration-manifest.csv
```

Expected structure:

```text
wrote 768 records to docs/experiments/results/hierarchical-calibration-manifest.csv
class counts: {'edited': 256, 'real': 256, 'synthetic': 256}
manifest SHA-256: 5a23bee4f802b9c88dd41030711aee173d27a3f8974371153439c0c363cde3a5
```

Perform a separate summary check:

```python
import csv
import hashlib
from collections import Counter
from pathlib import Path

path = Path(
    "/content/diffdetect/docs/experiments/results/"
    "hierarchical-calibration-manifest.csv"
)
rows = list(csv.DictReader(path.open(newline="")))

print("SHA-256:", hashlib.sha256(path.read_bytes()).hexdigest())
print("Rows:", len(rows))
print("Classes:", dict(Counter(row["true_class"] for row in rows)))
print("Sources:", dict(Counter(row["source_root"] for row in rows)))
print("Unique sample IDs:", len({row["sample_id"] for row in rows}))
print("Unique input hashes:", len({row["input_sha256"] for row in rows}))
print(
    "Exclusion manifests:",
    {row["exclusion_manifest_sha256"] for row in rows},
)
```

The expected values are 768 rows, 256 rows per class, 512 CocoGlide rows, 256
DiffusionDB rows, 768 unique sample identifiers, 768 unique input hashes, and
exactly one exclusion-manifest hash.

Do not run model inference yet. Preserve the generated CSV and its SHA-256 for
review and versioning.

The frozen manifest above was independently checked for class balance,
consecutive indexes, unique sample identifiers, unique image hashes,
deterministic ordering, exact CocoGlide calibration membership, and zero
sample, source, or content overlap with the previous validation manifest.

## 5. Start a clean GPU runtime for inference

After the manifest has been committed, start a clean T4 GPU runtime, mount
Drive again, and clone or update the experiment branch. Install the optional
runtime dependencies without replacing the Colab-provided PyTorch build:

```bash
python -m pip install -q \
  albumentations blobfile einops mpi4py numpy pandas pyarrow \
  PyYAML safetensors timm
```

Use a clean local Hugging Face cache. Reuse only the PyTorch cache in Drive:

```bash
export HF_HOME=/content/huggingface-hierarchical
export HF_HUB_CACHE=/content/huggingface-hierarchical/hub
export HF_HUB_DISABLE_XET=1
export TORCH_HOME=/content/drive/MyDrive/DiffDetect/cache/torch
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"
```

Clone the exact external revisions:

```bash
git clone https://github.com/miraflow/DistilDIRE.git /content/DistilDIRE
git -C /content/DistilDIRE checkout --detach \
  5de48cf255411b3f47833822e0afec32b2d66e61

git clone https://github.com/anonyme610/dinolizer.git /content/DinoLizer
git -C /content/DinoLizer checkout --detach \
  3241ce530a685e6e0560db4e0d8aaa9f28de6fc6
```

Point to the preserved model files:

```bash
export DISTILDIRE_CLASSIFIER=/content/drive/MyDrive/DiffDetect/models/distildire/imagenet-distil-dire-11e.pth
export DISTILDIRE_ADM=/content/drive/MyDrive/DiffDetect/models/distildire/256x256-adm.pt
export DINOLIZER_CHECKPOINT=/content/drive/MyDrive/DiffDetect/models/dinolizer/epoch=36_val_loss=0.2130.ckpt
```

The runner verifies repository revisions and every model digest before loading
a model.

## 6. Run the raw-evidence smoke test

The smoke test verifies all 768 inputs, then runs the earliest manifest image
from each class. It records raw evidence only; no hierarchical policy is fitted
or applied at this stage.

```bash
export RESULTS_ROOT=/content/drive/MyDrive/DiffDetect/results/hierarchical-calibration-v1
mkdir -p "$RESULTS_ROOT"

cd /content/diffdetect
python experiments/hierarchical_aggregation.py run \
  --manifest docs/experiments/results/hierarchical-calibration-manifest.csv \
  --cocoglide-root "$COCOGLIDE_CALIBRATION_ROOT" \
  --diffusiondb-root "$HIERARCHICAL_DIFFUSIONDB_ROOT" \
  --distildire-repository /content/DistilDIRE \
  --distildire-classifier "$DISTILDIRE_CLASSIFIER" \
  --distildire-adm "$DISTILDIRE_ADM" \
  --dinolizer-repository /content/DinoLizer \
  --dinolizer-checkpoint "$DINOLIZER_CHECKPOINT" \
  --device cuda \
  --environment-id colab-t4-hierarchical-calibration-v1-smoke \
  --output "$RESULTS_ROOT/hierarchical-calibration-smoke.csv" \
  --smoke
```

The command creates a three-row CSV and a metadata JSON. Both must report a
complete run, the frozen manifest hash, verified model revisions and hashes,
and `aggregation_policy_applied: false`. Stop and inspect these artifacts
before starting the 768-image inference.

## 7. Run the complete raw-evidence inference

Use `--resume` from the first attempt so every completed row remains reusable
after a runtime interruption:

```bash
python experiments/hierarchical_aggregation.py run \
  --manifest docs/experiments/results/hierarchical-calibration-manifest.csv \
  --cocoglide-root "$COCOGLIDE_CALIBRATION_ROOT" \
  --diffusiondb-root "$HIERARCHICAL_DIFFUSIONDB_ROOT" \
  --distildire-repository /content/DistilDIRE \
  --distildire-classifier "$DISTILDIRE_CLASSIFIER" \
  --distildire-adm "$DISTILDIRE_ADM" \
  --dinolizer-repository /content/DinoLizer \
  --dinolizer-checkpoint "$DINOLIZER_CHECKPOINT" \
  --device cuda \
  --environment-id colab-t4-hierarchical-calibration-v1 \
  --output "$RESULTS_ROOT/hierarchical-calibration.csv" \
  --resume
```

The frozen complete result contains 768 unique rows, no detector errors, and
has SHA-256
`f104b2ea9b08e25122382583e081a6216b079f26f30c358e6c660edd53f7f27b`.
Its metadata must report `status: complete` and
`aggregation_policy_applied: false`.

## 8. Fit and freeze the hierarchy offline

Copy the complete CSV and its metadata into `docs/experiments/results`, then
install the experiment-only dependency. A GPU is not used in this phase:

```bash
python -m pip install -e '.[experiments]'

python experiments/hierarchical_aggregation.py fit-policy \
  --results docs/experiments/results/hierarchical-calibration.csv \
  --results-metadata \
    docs/experiments/results/hierarchical-calibration.csv.metadata.json \
  --manifest \
    docs/experiments/results/hierarchical-calibration-manifest.csv \
  --output-dir docs/experiments/results
```

The fitter verifies the frozen input hashes, keeps every CocoGlide real/edited
pair in one of five grouped folds, standardizes features inside each training
fold, and selects both confidence thresholds exclusively from out-of-fold
predictions. It also fits the two prespecified descriptive ablations. The
earlier three-class validation result is not opened by this command.

The command writes the policy JSON, grouped fold assignments, out-of-fold
predictions, the complete threshold table, calibration metrics, and a checksum
list. Do not use `--overwrite` after reviewing and freezing these artifacts.

The reviewed policy has SHA-256
`c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187`.

## 9. Run the read-only retrospective comparison

Only after freezing the policy and its hash, apply it to the earlier result:

```bash
python experiments/hierarchical_aggregation.py evaluate-policy \
  --policy docs/experiments/results/hierarchical-policy.json \
  --policy-sha256 \
    c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187 \
  --baseline-results \
    docs/experiments/results/three-class-validation-clean-cache.csv \
  --baseline-metadata \
    docs/experiments/results/three-class-validation-clean-cache.csv.metadata.json \
  --output-dir docs/experiments/results
```

This command validates the frozen policy and baseline hashes, evaluates the
serialized coefficients with the pure-Python runtime, and performs the
prespecified 2,000-resample paired source-group bootstrap. It never imports
scikit-learn and contains no fitting path. Results and limitations are
documented in [`hierarchical-aggregation.md`](hierarchical-aggregation.md).

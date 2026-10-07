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
manifest SHA-256: <record this value>
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

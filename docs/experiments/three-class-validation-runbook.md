# Three-class validation runbook

This runbook prepares the frozen 768-image manifest and executes the shared
DistilDIRE plus DinoLizer smoke test in Google Colab. It implements the protocol
in [`three-class-validation-plan.md`](three-class-validation-plan.md).

Do not start model inference until the generated manifest has been reviewed and
committed. Changing a threshold, model artifact, input file, or decision rule
after that point creates a new protocol version.

## 1. Start a clean GPU runtime

Use a clean Colab GPU runtime, mount Drive, and clone this experiment branch:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
git clone --branch experiment/three-class-validation \
  https://github.com/JosueSucupira/diffdetect.git /content/diffdetect
cd /content/diffdetect
python -m pip install -e .
python -m pip install \
  albumentations blobfile einops mpi4py numpy pandas pyarrow \
  PyYAML safetensors timm
```

Use the PyTorch and torchvision versions already matched to the Colab CUDA
runtime. Do not downgrade them before the smoke test. The experiment records
their exact versions in the output metadata.

Use a clean local Hugging Face cache for the evaluated run. Keep the PyTorch
cache in Drive so the DINOv2 backbone can be reused, but do not point
`HF_HOME` at a partially synchronized Drive cache:

```bash
export HF_HOME=/content/huggingface-clean
export HF_HUB_CACHE=/content/huggingface-clean/hub
export HF_HUB_DISABLE_XET=1
export TORCH_HOME=/content/drive/MyDrive/DiffDetect/cache/torch
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"
```

The first complete attempt used a corrupted Hugging Face cache in Drive and
caused a safetensors `header too large` error in DinoLizer for every sample.
Those results were discarded. The local-cache configuration above was used
for the successful 768-image run. An unauthenticated Hub warning is harmless;
set `HF_TOKEN` only when a higher download rate limit is needed.

## 2. Prepare pinned external repositories

```bash
git clone https://github.com/miraflow/DistilDIRE.git /content/DistilDIRE
git -C /content/DistilDIRE checkout --detach 5de48cf

git clone https://github.com/anonyme610/dinolizer.git /content/DinoLizer
git -C /content/DinoLizer checkout --detach \
  3241ce530a685e6e0560db4e0d8aaa9f28de6fc6
```

The runner checks both repository revisions and all three model digests before
loading either model. Set the preserved checkpoint paths for the current Drive
layout:

```bash
export DISTILDIRE_CLASSIFIER=/content/drive/MyDrive/DiffDetect/models/distildire/imagenet-distil-dire-11e.pth
export DISTILDIRE_ADM=/content/drive/MyDrive/DiffDetect/models/distildire/256x256-adm.pt
export DINOLIZER_CHECKPOINT=/content/drive/MyDrive/DiffDetect/models/dinolizer/epoch=36_val_loss=0.2130.ckpt
```

Download the two DistilDIRE artifacts from their pinned official sources when
they are not already preserved in Drive, then verify the frozen digests before
continuing:

```bash
mkdir -p "$(dirname "$DISTILDIRE_CLASSIFIER")"

if [ ! -f "$DISTILDIRE_CLASSIFIER" ]; then
  wget -O "$DISTILDIRE_CLASSIFIER.part" \
    "https://huggingface.co/yevvonlim/distildire/resolve/1318c17f77ec70153365a3cd62d885a87c498e7d/imagenet-distil-dire-11e.pth?download=true"
  mv "$DISTILDIRE_CLASSIFIER.part" "$DISTILDIRE_CLASSIFIER"
fi

if [ ! -f "$DISTILDIRE_ADM" ]; then
  wget -O "$DISTILDIRE_ADM.part" \
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
  mv "$DISTILDIRE_ADM.part" "$DISTILDIRE_ADM"
fi

echo "e6b76a13ae49eb83d39fb9b1f7de86bf9f63d1dddf0225ca7ad03690e6bc53bc  $DISTILDIRE_CLASSIFIER" | sha256sum -c -
echo "fd9dd2335b8736d521de0aed54bd90ca  $DISTILDIRE_ADM" | md5sum -c -
```

If the DistilDIRE files use different directories in Drive, change only these
path variables. Do not rename or replace the files without verifying that they
have the frozen digests in the experiment plan.

## 3. Prepare the pinned DiffusionDB source

Download the fixed metadata and selected source archive. The archive contains
1,000 candidates; the manifest generator applies the frozen NSFW filter and
hash ranking to select 256.

```bash
export DIFFUSIONDB_REVISION=fb620fbe49fa4420e0734bd9c0df11f51176b61f
export DIFFUSIONDB_ROOT=/content/drive/MyDrive/DiffDetect/datasets/DiffusionDB/three-class-v1
mkdir -p "$DIFFUSIONDB_ROOT"

wget -O "$DIFFUSIONDB_ROOT/metadata.parquet" \
  "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/$DIFFUSIONDB_REVISION/metadata.parquet?download=true"
wget -O "$DIFFUSIONDB_ROOT/part-001079.zip" \
  "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/$DIFFUSIONDB_REVISION/images/part-001079.zip?download=true"
unzip -q "$DIFFUSIONDB_ROOT/part-001079.zip" -d "$DIFFUSIONDB_ROOT"
```

## 4. Generate the frozen manifest

Point `COCOGLIDE_ROOT` to the directory that directly contains files such as
`000000002592_real.png` and `000000002592_manipulated.png` from the preserved
CocoGlide `prepared-v1` validation data:

```bash
export COCOGLIDE_ROOT=/content/drive/MyDrive/DiffDetect/datasets/CocoGlide/prepared-v1/validation/images

cd /content/diffdetect
python experiments/three_class_validation.py prepare-manifest \
  --cocoglide-results docs/experiments/results/dinolizer-cocoglide-validation.csv \
  --cocoglide-root "$COCOGLIDE_ROOT" \
  --diffusiondb-metadata "$DIFFUSIONDB_ROOT/metadata.parquet" \
  --diffusiondb-root "$DIFFUSIONDB_ROOT" \
  --output docs/experiments/results/three-class-validation-manifest.csv
```

Expected output:

```text
wrote 768 records to docs/experiments/results/three-class-validation-manifest.csv
class counts: {'edited': 256, 'real': 256, 'synthetic': 256}
manifest SHA-256: 9dcda4b93d8c73df7b1240629bec209566cd6119bb7f800c2906b8c3826b8777
```

At this point, stop before inference. Copy or commit the generated manifest on
the experiment branch so its contents and SHA-256 can be reviewed and frozen.

## 5. Execute the smoke test after manifest review

The smoke test selects the earliest manifest entry from each true class. It
still verifies all 768 input files before loading the models.

```bash
export RESULTS_ROOT=/content/drive/MyDrive/DiffDetect/results/three-class-v1
mkdir -p "$RESULTS_ROOT"

cd /content/diffdetect
python experiments/three_class_validation.py run \
  --manifest docs/experiments/results/three-class-validation-manifest.csv \
  --cocoglide-root "$COCOGLIDE_ROOT" \
  --diffusiondb-root "$DIFFUSIONDB_ROOT" \
  --distildire-repository /content/DistilDIRE \
  --distildire-classifier "$DISTILDIRE_CLASSIFIER" \
  --distildire-adm "$DISTILDIRE_ADM" \
  --dinolizer-repository /content/DinoLizer \
  --dinolizer-checkpoint "$DINOLIZER_CHECKPOINT" \
  --device cuda \
  --environment-id colab-t4-three-class-v1-smoke \
  --output "$RESULTS_ROOT/three-class-smoke.csv" \
  --smoke
```

The command creates:

- `three-class-smoke.csv`, with one real, one synthetic, and one edited row;
- `three-class-smoke.csv.metadata.json`, with software versions, verified
  artifact identities, status, and GPU memory information.

The first row includes lazy loading of both models. Later rows reuse them. Do
not interpret three smoke classifications as performance evidence; this phase
only checks shared-environment compatibility and output integrity.

If Colab interrupts a later complete run, repeat the same command with
`--resume`. Existing sample identifiers are verified and skipped; errors and
conflicts already written to the CSV are not silently retried.

## 6. Recorded result

The completed evaluation, derived metrics, figures, interpretation, and exact
artifact hashes are recorded in
[`three-class-validation.md`](three-class-validation.md).

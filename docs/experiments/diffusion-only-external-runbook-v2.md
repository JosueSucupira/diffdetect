# Runbook de Execução — Reavaliação Externa de Difusão (v2)

Este guia contém as células prontas para execução no **Google Colab com GPU T4**, avaliando o conjunto de 384 imagens 100% baseadas em modelos de difusão.

---

## Célula 1: Montar Drive e Preparar Ambiente

```python
# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar ou atualizar o repositório na branch de teste
import os
%cd /content
if not os.path.exists('diffdetect'):
    !git clone https://github.com/JosueSucupira/diffdetect.git
%cd /content/diffdetect
!git fetch origin
!git checkout experiment/pure-diffusion-confirmation
!git pull origin experiment/pure-diffusion-confirmation

# 3. Instalar dependências necessárias
!pip install -q -e .
!pip install -q scikit-learn timm Pillow
```

---

## Célula 2: Download Automático das 64 Imagens do Synthbuster

Esta célula baixa apenas as 64 imagens selecionadas de forma determinística do Hugging Face (`Irma-5/Synthbuster_compressed`): 32 de *Midjourney v5* e 32 de *Stable Diffusion XL*, sem sobreposição com o RAISE-1k.

```python
import os, urllib.request, hashlib

out_base = "/content/datasets/synthbuster/images"
os.makedirs(f"{out_base}/midjourney-v5", exist_ok=True)
os.makedirs(f"{out_base}/stable-diffusion-xl", exist_ok=True)

# Lista determinística de imagens do manifesto v2
import csv
manifest_path = "docs/experiments/results/diffusion-only-v2-manifest.csv"
with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    synth_samples = [r for r in reader if r["source_dataset"] == "Synthbuster"]

print(f"Baixando {len(synth_samples)} imagens do Synthbuster...")

base_url = "https://huggingface.co/datasets/Irma-5/Synthbuster_compressed/resolve/main/orig/fake"

for row in synth_samples:
    rel_path = row["relative_path"].removeprefix("images/")
    target_path = os.path.join(out_base, rel_path)
    
    if not os.path.exists(target_path):
        # Baixar do Hugging Face
        url = f"{base_url}/{rel_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_f:
            out_f.write(resp.read())

    # Verificar hash de integridade
    with open(target_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == row["input_sha256"], f"Hash incorreto para {rel_path}!"

print("Download concluído com 100% de integridade verificada!")
```

---

## Célula 3: Executar a Inferência com os Detectores (DistilDIRE + DinoLizer)

Esta célula roda a inferência nos 384 exemplos (duração aproximada: 10 a 12 minutos na GPU T4).

```python
import os

# Ajuste os caminhos abaixo conforme a estrutura do seu Google Drive se necessário:
DRIVE_DIR = "/content/drive/MyDrive/DiffDetect"

# Raízes dos datasets
ROOT_OPENIMAGES = f"{DRIVE_DIR}/datasets/external-v1/openimages-v6"
ROOT_RAISE = f"{DRIVE_DIR}/datasets/external-v1/raise-1k"
ROOT_MAGICBRUSH = f"{DRIVE_DIR}/datasets/external-v1/magicbrush"
ROOT_ABSTRACTEDIT = f"{DRIVE_DIR}/datasets/external-v1/abstractedit"
ROOT_GENIMAGE = f"{DRIVE_DIR}/datasets/external-v1/genimage"
ROOT_SYNTHBUSTER = "/content/datasets/synthbuster"

# Pesos e repositórios
DISTILDIRE_REPO = f"{DRIVE_DIR}/repos/DistilDIRE"
DISTILDIRE_CLASSIFIER = f"{DRIVE_DIR}/models/distildire/imagenet-distil-dire-11e.pth"
DISTILDIRE_ADM = f"{DRIVE_DIR}/models/distildire/256x256-adm.pt"

DINOLIZER_REPO = f"{DRIVE_DIR}/repos/dinolizer"
DINOLIZER_CHECKPOINT = f"{DRIVE_DIR}/models/dinolizer/epoch=36_val_loss=0.2130.ckpt"

OUT_EVIDENCE = f"{DRIVE_DIR}/results/diffusion-only-v2-evidence.csv"
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)

# Executar inferência oficial
!PYTHONPATH=src python experiments/hierarchical_external_validation.py run \
  --manifest docs/experiments/results/diffusion-only-v2-manifest.csv \
  --root openimages-v6={ROOT_OPENIMAGES} \
  --root raise-1k={ROOT_RAISE} \
  --root magicbrush={ROOT_MAGICBRUSH} \
  --root abstractedit={ROOT_ABSTRACTEDIT} \
  --root genimage={ROOT_GENIMAGE} \
  --root synthbuster={ROOT_SYNTHBUSTER} \
  --distildire-repository {DISTILDIRE_REPO} \
  --distildire-classifier {DISTILDIRE_CLASSIFIER} \
  --distildire-adm {DISTILDIRE_ADM} \
  --dinolizer-repository {DINOLIZER_REPO} \
  --dinolizer-checkpoint {DINOLIZER_CHECKPOINT} \
  --device cuda \
  --environment-id colab-t4-diffusion-only-v2 \
  --output {OUT_EVIDENCE}
```

---

## Célula 4: Avaliar a Política Hierárquica e Visualizar Métricas

```python
import os, json

DRIVE_DIR = "/content/drive/MyDrive/DiffDetect"
OUT_EVIDENCE = f"{DRIVE_DIR}/results/diffusion-only-v2-evidence.csv"
OUT_META = f"{DRIVE_DIR}/results/diffusion-only-v2-evidence.csv.metadata.json"
OUT_DIR = "docs/experiments/results/diffusion-only-v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Rodar avaliação com a política hierárquica congelada
!PYTHONPATH=src python experiments/hierarchical_external_validation.py evaluate \
  --policy docs/experiments/results/hierarchical-policy.json \
  --policy-sha256 c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187 \
  --manifest docs/experiments/results/diffusion-only-v2-manifest.csv \
  --results {OUT_EVIDENCE} \
  --results-metadata {OUT_META} \
  --output-dir {OUT_DIR} \
  --overwrite

# Exibir relatório resumo
metrics_file = f"{OUT_DIR}/hierarchical-external-metrics.json"
if os.path.exists(metrics_file):
    with open(metrics_file) as f:
        m = json.load(f)
    
    print("\n" + "="*50)
    print("           RESULTADOS DIFFUSION-ONLY (v2)")
    print("="*50)
    hier = m["hierarchical"]["end_to_end"]
    print(f"Acurácia Geral:      {hier['accuracy']*100:.2f}%")
    print(f"Acurácia Balanceada: {hier['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1:            {hier['macro_f1']*100:.2f}%")
    print(f"Cobertura:           {hier['coverage']*100:.2f}%")
    print(f"Taxa de Abstenção:   {hier['abstention_rate']*100:.2f}%")
    print("-" * 50)
    print("Recall por Classe:")
    for cls_name, rec in hier["per_class"].items():
        print(f"  - {cls_name.capitalize():10s}: Recall = {rec['recall']*100:.2f}% | F1 = {rec['f1']*100:.2f}%")
    print("-" * 50)
    print("Critérios de Aceitação:")
    for gate, status in m["confirmation_acceptance"].items():
        print(f"  - {gate}: {'APROVADO' if status else 'REPROVADO'}")
    print("="*50)
```

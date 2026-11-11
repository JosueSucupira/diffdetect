# Reavaliação Externa de Escopo Restrito — Modelos de Difusão (v2)

Este protocolo documenta a reavaliação externa independente do framework **DiffDetect**, restrita estritamente a **imagens geradas e manipuladas por modelos de difusão comprovados**.

## 1. Contexto e Motivação

O experimento externo original (`diffdetect-hierarchical-external-confirmation-v1`), documentado em [`hierarchical-external-confirmation.md`](hierarchical-external-confirmation.md), continha 64 imagens da fonte *AI Detector Arena v0.1* (32 do GPT Image 1.5 e 32 do Gemini 3 Pro). Essa inclusão emergencial ocorreu após a indisponibilidade temporária do repositório Zenodo do *Synthbuster* em 08/10/2026.

Contudo, como o tema central e delimitador da pesquisa de TCC é a **detecção de modelos de difusão**, modelos proprietários fechados ou de arquitetura desconhecida violam a premissa metodológica da tese. Além disso, o detector base **DistilDIRE** opera por meio do *Diffusion Reconstruction Error* (DIRE), que avalia especificamente a trajetória reversa de difusão.

Este protocolo v2 restaura o plano original da pesquisa, incorporando as partições legítimas de difusão do **Synthbuster** (espelhadas publicamente no Hugging Face) em substituição aos modelos de fronteira fechados.

## 2. Alocação Estratificada do Corpus (384 imagens)

A distribuição é estritamente balanceada (128 imagens por classe verdadeira):

| Classe | Fonte do Dataset | Subconjunto / Método | Quantidade | Descrição Técnica |
| :--- | :--- | :--- | :---: | :--- |
| **Real** | Open Images V6 validation | Amostragem determinística por hash | 64 | Fotografias autênticas do mundo real |
| **Real** | RAISE-1k (Synthbuster) | Amostragem determinística por hash | 64 | Fotografias puras de câmeras digitais (TIFF/PNG descompactado) |
| **Sintética** | GenImage validation | VQDM | 32 | Vector Quantized Diffusion Model |
| **Sintética** | GenImage validation | Wukong | 32 | LDM / Stable Diffusion treinado em domínio oriental |
| **Sintética** | Synthbuster (`Irma-5/Synthbuster_compressed`) | Midjourney v5 | 32 | Modelo de difusão comercial de alta fidelidade visual |
| **Sintética** | Synthbuster (`Irma-5/Synthbuster_compressed`) | Stable Diffusion XL | 32 | Modelo de difusão latente moderno de alta resolução |
| **Editada** | MagicBrush public dev | DALL-E 2 (1º turno) | 64 | Inpainting localizado via difusão hierárquica (unCLIP) |
| **Editada** | AbstractEdit | Flux-Kontext | 64 | Edição contínua via Flow Matching / Rectified Flow |

## 3. Independência e Controle de Vazamento

1. **Exclusão de experimentos anteriores**: Nenhuma imagem ou hash presente em `three-class-validation-manifest.csv` ou `hierarchical-calibration-manifest.csv` foi aceita.
2. **Independência entre RAISE-1k e Synthbuster**: O Synthbuster utiliza imagens base do RAISE para formular seus prompts. Das 1.000 imagens sintéticas do Midjourney v5 e SDXL, foram **excluídos todos os 64 identificadores já utilizados na partição real do RAISE-1k**, garantindo 100% de independência conceitual e ausência de imagens pareadas entre as classes real e sintética no teste externo.
3. **Quotas de Diversidade**:
   - Cada classe possui exatamente duas fontes de datasets principais ($\le 50\%$ de share por dataset);
   - A classe sintética possui 4 métodos distintos ($\le 25\%$ de share por método: VQDM, Wukong, Midjourney v5, SDXL).

## 4. Hashes dos Artefatos Congelados

- **Catálogo de candidatos (v2)**: `docs/experiments/results/diffusion-only-v2-candidate-catalog.csv`  
  SHA-256: `ea7b9b7f3cee6ffaebc7ea5f090d3b7ab095c1d8af8e34e0dc4948ce8a565791`
- **Manifesto de execução (v2)**: `docs/experiments/results/diffusion-only-v2-manifest.csv`  
  SHA-256: `f3624b140bdcf486409953c74ed4d65bab25c78ff5c6a43d3287d5c7df242d33`

## 5. Metas Científicas Pré-estabelecidas

Mantêm-se os mesmos critérios de aceitação formalmente estabelecidos para a confirmação hierárquica:

- Erros de agregação: **0**
- Cobertura de decisão: $\ge \mathbf{80\%}$
- Acurácia balanceada: $\ge \mathbf{70\%}$
- Macro F1: $\ge \mathbf{65\%}$
- Recall de cada classe: $\ge \mathbf{50\%}$

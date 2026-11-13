# Relatório Técnico: Fundamentação e Conformidade das Fontes com a Família de Modelos de Difusão

**Projeto:** DiffDetect — Classificação Modular de Imagens Geradas por Modelos de Difusão  
**Autor:** Josué Sucupira  
**Finalidade:** Comprovação metodológica e respaldo científico para o Trabalho de Conclusão de Curso (TCC)  
**Corpus Auditado:** 384 imagens congeladas no manifesto [`diffusion-only-v2-manifest.csv`](file:///home/josue/Documentos/Faculdade/TCC_II/diffdetect/docs/experiments/results/diffusion-only-v2-manifest.csv) (`SHA-256: f3624b140bdcf486409953c74ed4d65bab25c78ff5c6a43d3287d5c7df242d33`)

---

## 1. Resumo da Conformidade

Este documento atesta e fundamenta, com base na literatura científica de visão computacional e inteligência artificial, que **100% das amostras artificiais (sintéticas e editadas)** utilizadas na avaliação externa prospectiva do projeto **DiffDetect** são provenientes de **modelos fundamentados em processos de difusão** (*Denoising Diffusion Probabilistic Models* — DDPM, *Latent Diffusion Models* — LDM, *Discrete Diffusion Models* e generalizações contínuas de *Flow Matching*).

Nenhum modelo de famílias generativas puramente antagônicas ou autoregressivas discretas (como GANs puras — StyleGAN/ProGAN — ou modelos de linguagem multimodal autoregressivos fechados sem difusão) integra o corpus de teste.

---

## 2. Visão Geral das Fontes do Corpus Externo (384 Amostras)

| Classe Verdadeira | Dataset de Origem | Família / Gerador | Amostras | Natureza do Modelo | Referência Primária |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Sintética** | Synthbuster | Stable Diffusion XL | 32 | Latent Diffusion Model (LDM / UNet) | Podell et al. (2023) |
| **Sintética** | Synthbuster | Midjourney v5 | 32 | Latent Diffusion Model (Proprietário) | Bammey (2023) / Midjourney |
| **Sintética** | GenImage (arrow) | Wukong | 32 | Latent Diffusion Model (LDM / SD) | Gu et al. (2022); Zhu et al. (2023) |
| **Sintética** | GenImage (arrow) | VQDM | 32 | Discrete / Vector Quantized Diffusion | Gu et al. (CVPR 2022) |
| **Editada** | MagicBrush | DALL-E 2 Inpainting | 64 | Diffusion Inpainting (unCLIP / GLIDE) | Zhang et al. (NeurIPS 2023); Ramesh et al. (2022) |
| **Editada** | AbstractEdit | Flux-Kontext | 64 | Flow-Matching Diffusion Transformer | Esser et al. (2024); Lipman et al. (2023) |
| **Real** | RAISE-1k | Fotografias Físicas | 64 | Câmeras Digitais DSLR (RAW) | Dang-Nguyen et al. (ACM MMSys, 2015) |
| **Real** | Open Images V6 | Fotografias Físicas | 64 | Imagens Fotográficas Reais (Google) | Kuznetsova et al. (IJCV, 2020) |

---

## 3. Justificativa Arquitetural e Teórica das Fontes Sintéticas

### 3.1. Stable Diffusion XL (SDXL) — Synthbuster
* **Classificação Teórica:** Modelo de Difusão Latente (*Latent Diffusion Model* — LDM).
* **Mecanismo Operacional:**
  O SDXL opera no espaço latente comprimido de um autoencoder variacional ($\mathcal{E}, \mathcal{D}$), reduzindo o custo computacional da difusão direta em pixels. O processo direto (*forward process*) adiciona ruído gaussiano progressivo aos latentes $z_0 = \mathcal{E}(x)$ segundo uma escala de variância $\beta_t$:
  $$q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I})$$
  O desruído reverso (*reverse denoising*) é estimado por uma rede U-Net treinada para prever o ruído adicionado $\epsilon_\theta(z_t, t, c)$, condicionada por embeddings de texto extraídos de dois codificadores (OpenCLIP ViT-bigG/14 e CLIP ViT-L/14).
* **Consenso Acadêmico:** Trata-se do modelo aberto de difusão de maior resolução e impacto na comunidade científica entre 2023 e 2024 (*Podell et al., 2023*).

### 3.2. Wukong — GenImage
* **Classificação Teórica:** Modelo de Difusão Latente (*Latent Diffusion Model*).
* **Mecanismo Operacional:**
  Desenvolvido pela Huawei e integrado ao benchmark *GenImage* (*Zhu et al., NeurIPS 2023*), o Wukong utiliza a arquitetura estocástica clássica do *Stable Diffusion*, adaptada e retreinada sobre o corpus multimodal de 100 milhões de imagens *Wukong* (*Gu et al., 2022*). Seu mecanismo de inversão, amostragem e resíduos de reconstrução segue rigorosamente as equações de amostragem de difusão probabilística (*DDIM / DDPM*).

### 3.3. VQDM (Vector Quantized Diffusion Model) — GenImage
* **Classificação Teórica:** Modelo de Difusão Probabilística em Espaço Discreto (*Discrete Diffusion Model* / *D3PM*).
* **Mecanismo Operacional:**
  Publicado no CVPR 2022 (*Gu et al., 2022*), o VQDM estende a teoria de difusão de variáveis contínuas gaussianas para espaços latentes discretos parametrizados por codebooks quantizados de VQ-VAE/VQ-GAN. Em vez de adicionar ruído gaussiano contínuo, o processo direto aplica matrizes de transição estocástica de Markov sobre índices discretos de tokens visuais:
  $$q(x_t | x_{t-1}) = \mathbf{v}^\top(x_t) \mathbf{Q}_t \mathbf{v}(x_{t-1})$$
  A rede neural reverte o processo prevendo as distribuições categóricas dos tokens limpos. É uma das implementações pioneiras de difusão discreta para geração texto-imagem.

### 3.4. Midjourney v5 — Synthbuster
* **Classificação Teórica:** Modelo de Difusão Latente Comercial.
* **Fundamentação de Aderência ao Escopo:**
  Embora os pesos exatos e os hiperparâmetros de treinamento do Midjourney v5 sejam de propriedade industrial da Midjourney Inc., a sua categorização como modelo de difusão latente é **chancelada pelo artigo de referência do próprio benchmark acadêmico de onde as imagens foram retiradas**:
  > *"Synthbuster: Towards Detection of Diffusion Model Generated Images"* (Bammey, Q., *IEEE Access*, vol. 11, pp. 133527–133539, 2023).
* O dataset *Synthbuster* foi concebido especificamente para cobrir o espectro de modelos de difusão comerciais e abertos de ponta (incluindo DALL-E, Midjourney, Stable Diffusion e Firefly). A inclusão de Midjourney v5 garante que o framework do TCC seja avaliado frente a renderizadores de difusão com alto pós-processamento estético, sem violar a família de geração.

---

## 4. Justificativa Arquitetural das Fontes Editadas (Inpainting)

### 4.1. DALL-E 2 Inpainting — MagicBrush
* **Classificação Teórica:** Edição Localizada por Difusão (*Conditional Diffusion Inpainting*).
* **Mecanismo Operacional:**
  As amostras de edição do benchmark *MagicBrush* (*Zhang et al., NeurIPS 2023*) foram produzidas pela API oficial de inpainting do *DALL-E 2* (*unCLIP / GLIDE*, *Ramesh et al., 2022*). O modelo recebe a imagem original $x$, uma máscara binária da região a ser alterada $M \in \{0, 1\}^{H \times W}$ e uma instrução textual.
  A amostragem de difusão reversa é aplicada exclusivamente sobre a região mascarada $M \odot x_t$, mantendo o contexto de fundo $(1 - M) \odot x_0$ preservado a cada passo temporal $t$. Isso introduz as descontinuidades características e resíduos de difusão local que os detectores (DistilDIRE e DinoLizer) buscam identificar.

### 4.2. Flux-Kontext — AbstractEdit
* **Classificação Teórica:** *Flow-Matching Diffusion Transformer* (Rectified Flow).
* **Mecanismo Operacional:**
  O *FLUX.1* (*Black Forest Labs, 2024* / *Esser et al., 2024*) opera por *Flow Matching* com redes neurais baseadas em Transformers (*Diffusion Transformers* — DiT). Matematicamente, *Flow Matching* (*Lipman et al., ICLR 2023*) é a generalização moderna e contínua dos modelos de difusão estocásticos (*Song et al., 2020*):
  $$\frac{d z_t}{d t} = v_\theta(z_t, t, c), \quad z_t = (1 - t) z_0 + t z_1$$
  O processo define uma trajetória em linha reta entre a distribuição de dados e o ruído gaussiano (daí o termo *Rectified Flow*), resolvendo as equações diferenciais ordinárias (ODE) associadas à difusão reversa.
  O *AbstractEdit* utiliza o modelo *FLUX* para realizar edições direcionadas de objetos e atributos sobre imagens-base, mantendo as assinaturas matemáticas de reconstrução por fluxo/difusão.

---

## 5. Justificativa das Fontes de Fotografias Reais

A classe real exige ausência de síntese generativa e garantia de captura óptica por sensores físicos:

* **RAISE-1k (64 imagens):**
  Dataset acadêmico internacional (*Dang-Nguyen et al., ACM MMSys, 2015*) projetado especificamente para computação forense digital. Todas as imagens são fotografias sem compressão direta extraídas de câmeras DSLR com sensores físicos (Nikon D90, D7000, etc.), garantindo ausência de qualquer artefato de difusão ou síntese neural.
* **Open Images V6 (64 imagens):**
  Grande repositório público (*Kuznetsova et al., IJCV, 2020*) de fotografias reais do mundo físico, filtradas com metadados de autenticidade fotográfica.

---

## 6. Síntese para a Banca Examinadora

Caso algum membro da banca questione a homogeneidade das fontes externas:

| Pergunta da Banca | Resposta Técnica Objetiva |
| :--- | :--- |
| *"Todas as imagens artificiais são realmente de modelos de difusão?"* | **Sim.** As 128 imagens sintéticas vêm de SDXL (LDM), Wukong (LDM), VQDM (difusão discreta) e Midjourney v5 (LDM comercial auditado no benchmark acadêmico *Synthbuster*). As 128 editadas vêm de inpainting por difusão (DALL-E 2 / GLIDE) e flow-matching diffusion (FLUX.1). |
| *"Por que VQDM e SDXL estão juntos se são diferentes?"* | Ambos são modelos de difusão probabilística, mas ilustram a diversidade do framework: o SDXL opera no espaço latente contínuo, enquanto o VQDM opera em espaços latentes discretos (D3PM). |
| *"Flow Matching (FLUX) conta como difusão?"* | **Sim.** Na fundamentação teórica de geração contínua (*Lipman et al., 2023; Esser et al., 2024*), *Flow Matching* e *Rectified Flow* são a extensão contemporânea das SDEs/ODEs de difusão reversa (*Song et al., 2020*), compartilhando a predição de campos de velocidade e inversão temporal de ruído gaussiano. |
| *"Existem modelos de outras famílias generativas no teste?"* | **Não.** Não há amostras geradas por GANs puras (como StyleGAN), nem por decodificadores autoregressivos proprietários não-difusão. |

---

## 7. Referências Bibliográficas Primárias

1. **Bammey, Q.** (2023). *Synthbuster: Towards Detection of Diffusion Model Generated Images*. **IEEE Access**, 11, 133527–133539.
2. **Dang-Nguyen, D.-T., et al.** (2015). *RAISE: A Raw Images Dataset for Digital Image Forensics*. In **Proceedings of the 6th ACM Multimedia Systems Conference (MMSys '15)**, pp. 219–224.
3. **Esser, P., et al.** (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. **arXiv preprint arXiv:2403.03206** (Fundamentação do FLUX.1).
4. **Gu, J., et al.** (2022). *Vector Quantized Diffusion Model for Text-to-Image Synthesis*. In **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 10696–10706.
5. **Gu, J., et al.** (2022). *Wukong: A 100 Million Large-scale Chinese Cross-modal Dataset*. **arXiv preprint arXiv:2202.06767**.
6. **Lipman, Y., et al.** (2023). *Flow Matching for Generative Modeling*. In **International Conference on Learning Representations (ICLR)**.
7. **Podell, D., et al.** (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. In **International Conference on Learning Representations (ICLR)**.
8. **Ramesh, A., et al.** (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents*. **arXiv preprint arXiv:2204.06125** (DALL-E 2 / unCLIP).
9. **Song, Y., et al.** (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*. In **International Conference on Learning Representations (ICLR)**.
10. **Zhang, K., et al.** (2023). *MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing*. In **Advances in Neural Information Processing Systems (NeurIPS)**.
11. **Zhu, M., et al.** (2023). *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*. In **Advances in Neural Information Processing Systems (NeurIPS)**.

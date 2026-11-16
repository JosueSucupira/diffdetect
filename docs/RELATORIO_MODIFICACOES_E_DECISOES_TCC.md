# Relatório Consolidado de Modificações e Decisões Técnicas — DiffDetect (TCC II)

**Projeto:** DiffDetect — Classificação Modular de Imagens Reais, Sintéticas e Editadas por Modelos de Difusão  
**Autor:** Josué Sucupira  
**Data:** 16 de Novembro de 2026  
**Repositório:** `diffdetect` (`git@github.com:JosueSucupira/diffdetect.git`)  
**Branch Principal:** `main` (commit consolidado `ccc201a`)  
**Status dos Testes:** 69 testes aprovados (100% de sucesso)

---

## 1. Visão Geral e Contexto das Decisões

Durante o ciclo de alinhamento com a orientação do Trabalho de Conclusão de Curso (TCC II), foram identificados quatro pontos críticos que demandavam ajustes conceituais, metodológicos e de código:

1. **Eliminação da Política Booleana:** O baseline booleano inicial combinava decisões binárias independentes e gerava 59,38% de conflitos (6,64% de acurácia balanceada). Em consenso com o orientador, decidiu-se **remover integralmente a regra booleana** do repositório, dos testes e da documentação, consolidando o *DiffDetect* exclusivamente em torno da sua arquitetura hierárquica.
2. **Resolução de Decisões Inconclusivas e Acúmulo de Erro:** A política hierárquica em cascata mantinha uma zona de abstenção no Nível 1 ($p_{\text{artificial}} \in [0.27, 0.73]$), que produzia predições inconclusivas e propagava erros de forma irreversível para o Nível 2. Implementou-se a **Regra Conjunta de Bayes (*Maximum A Posteriori*)**, atingindo **100% de cobertura** e elevando a acurácia balanceada para **78,91%**.
3. **Auditoria e Calibração dos Detectores (DistilDIRE e DinoLizer):** Investigou-se o código oficial e os artigos dos modelos integrados para alinhar os limiares nominais aos padrões da literatura (fixando o DistilDIRE em `0.5`, padrão oficial ICML 2024, e confirmando o limiar de pixel `0.5` do DinoLizer).
4. **Fundamentação Conceitual do Desempenho da Classe Editada:** Investigou-se a causa do recall de 66,41% na classe editada, demonstrando quantitativamente que edições cirúrgicas (*inpainting* via DALL-E 2) alcançam **84,38% de recall**, enquanto edições contextuais de larga escala (Flux-Kontext) alteram ~48% dos pixels da imagem e são fisicamente interpretadas pelo classificador como síntese integral (com 98,44% das imagens detectadas como IA).

---

## 2. Detalhamento das Modificações no Repositório

### 2.1. Implementação da Regra Conjunta de Bayes (`src/diffdetect/hierarchical.py`)

A arquitetura original avaliava os dois níveis sequencialmente: se o Nível 1 não atingisse certeza de 73%, a decisão era inconclusiva; se encaminhada para o Nível 2, uma classificação errônea no Nível 1 não podia mais ser corrigida.

**Solução Implementada:**
Adicionou-se ao runtime em Python puro a opção de decisão conjunta (`decision_rule="joint"`). O classificador calcula a probabilidade conjunta das três classes a partir das duas regressões logísticas calibradas:

$$\begin{aligned}
P(\text{real}) &= 1 - P(\text{artificial}) \\
P(\text{sintética}) &= P(\text{artificial}) \times P(\text{sintética} \mid \text{artificial}) \\
P(\text{editada}) &= P(\text{artificial}) \times [1 - P(\text{sintética} \mid \text{artificial})]
\end{aligned}$$

Propriedades matemáticas garantidas:
- $P(\text{real}) + P(\text{sintética}) + P(\text{editada}) = 1,00$ em todas as inferências.
- A predição final maximiza a probabilidade *a posteriori*:
  $$\hat{y} = \arg\max_{c \in \{\text{real}, \text{sintética}, \text{editada}\}} P(c)$$
- A classe de resultado (`HierarchicalClassificationResult` e `DiffDetectResult`) passou a expor `p_real`, `p_synthetic`, `p_edited` e `decision_rule`, garantindo total explicabilidade ao usuário final.

### 2.2. Calibragem do DistilDIRE (`src/diffdetect/adapters/distildire.py`, `factory.py`)

- **Investigação:** No repositório oficial do DistilDIRE (Wang et al., ICML 2024), a predição binária é expressa por $\sigma(y) > 0,5$. No *DiffDetect*, o limiar padrão nominal estava registrado provisoriamente como `0.987781...` devido a calibrações restritas em subconjuntos externos.
- **Ação:** O valor padrão `DEFAULT_DISTILDIRE_THRESHOLD` foi redefinido para `0.5`, em estrita conformidade com o artigo original.
- **Nota técnica:** A política hierárquica consome o score contínuo do detector padronizado por Z-score, e não a decisão booleana. Essa correção alinhou o contrato do adaptador isolado com a literatura sem alterar os coeficientes do modelo hierárquico.

### 2.3. Remoção Completa da Política Booleana

Para cumprir a determinação do orientador e eliminar a abordagem booleana legada do projeto:

1. **Núcleo de Classificação (`src/diffdetect/`):**
   - `src/diffdetect/classification.py`: Removidas as funções `classify()`, `ClassificationResult` e a enumeração `DecisionReason`. O arquivo agora mantém exclusivamente o enum oficial `ImageClass` (`REAL`, `SYNTHETIC`, `EDITED`).
   - `src/diffdetect/__init__.py`: Removidas as exportações de `classify`, `ClassificationResult` e `DecisionReason`.
2. **Exclusão de Scripts e Testes Legados:**
   - `experiments/three_class_validation.py` (Deletado — 1.058 linhas removidas)
   - `experiments/hierarchical_retrospective.py` (Deletado — 632 linhas removidas)
   - `tests/test_classification.py` (Deletado — 90 linhas removidas)
   - `tests/test_hierarchical_retrospective.py` (Deletado — 84 linhas removidas)
   - `tests/test_three_class_validation_experiment.py` (Deletado — 361 linhas removidas)
3. **Desacoplamento de Utilitários:**
   - As funções necessárias de extração de evidências e métricas (`_runs_from_row`, `_coverage_by_class`, `_stage_metrics`) foram movidas diretamente para `experiments/hierarchical_external_validation.py`.
   - Removido o subcomando `evaluate-policy` de `experiments/hierarchical_aggregation.py`.
4. **Remoção de Artefatos em Disco:**
   - Deletados 17 arquivos legados de cache, manifestos, figuras e métricas JSON em `docs/experiments/results/` (`three-class-validation-*` e `hierarchical-retrospective-*`).
5. **Limpeza de Documentação (`.md`):**
   - `README.md`: Atualizado para refletir o classificador hierárquico moderno, removendo referências à função booleana.
   - Planos e runbooks (`hierarchical-aggregation-plan.md`, `hierarchical-aggregation-runbook.md`, `hierarchical-external-confirmation-plan.md`, etc.): Limpos de todas as referências ao baseline booleano e à comparação retrospectiva.

---

## 3. Síntese dos Resultados Experimentais

### 3.1. Comparativo: Cascata Anterior vs. Regra Conjunta de Bayes

Avaliação sobre o corpus externo congelado de **384 imagens puras de modelos de difusão** (128 reais, 128 sintéticas e 128 editadas, provenientes de RAISE-1k, Open Images V6, SDXL, Midjourney v5, Wukong, VQDM, DALL-E 2 e Flux-Kontext):

| Métrica | Cascata Anterior | Regra Conjunta Atual | Variação Absoluta |
| :--- | :---: | :---: | :---: |
| **Cobertura Operacional** | 91,41% (351/384) | **100,00% (384/384)** | **+8,59 pp** |
| **Decisões Inconclusivas** | 33 imagens (8,59%) | **0 imagens (0,00%)** | **-33 imagens** |
| **Acurácia Balanceada Global** | 75,78% | **78,91%** | **+3,13 pp** |
| **Macro F1** | 77,25% | **78,75%** | **+1,50 pp** |
| **Recall — Imagens Reais** | 93,75% (120/128) | **94,53% (121/128)** | **+0,78 pp** |
| **Recall — Imagens Sintéticas** | 67,97% (87/128) | **75,78% (97/128)** | **+7,81 pp** |
| **Recall — Imagens Editadas** | 65,62% (84/128) | **66,41% (85/128)** | **+0,79 pp** |
| **Acertos Totais** | 291 / 384 | **303 / 384** | **+12 imagens** |

### 3.2. Matriz de Confusão Definitiva (Regra Conjunta)

Sem nenhuma decisão inconclusiva, a matriz $3 \times 3$ apresenta a seguinte distribuição exata:

```text
                 Predição
Verdadeiro     Real   Sintética  Editada   Total   Recall
Real           121         0        7       128    94,53%
Sintética        2        97       29       128    75,78%
Editada          9        34       85       128    66,41%
Total          132       131      121       384    
Precisão     91,67%   74,05%   70,25%            78,91% (Bal. Acc)
```

- **Excelente rejeição de falsos positivos:** Apenas 7 fotografias reais foram associadas a edição e nenhuma a síntese integral.
- **Detecção de inteligência artificial:** No Nível 1 (Real vs. Artificial), a acurácia é de **92,45%** (355 de 384 corretas) e a acurácia balanceada é de **91,02%**. Quase todas as confusões remanescentes ocorrem entre *sintética* e *editada*.

---

## 4. Diagnóstico Detalhado do Recall da Classe Editada (66,41%)

A investigação das 128 imagens editadas revelou que a classe é composta por dois perfis tecnológicos radicalmente diferentes:

| Método / Dataset | Natureza da Modificação | Área Modificada Média | Acertos (`edited`) | Confundidas com `synthetic` | Confundidas com `real` | Recall Específico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DALL-E 2** (MagicBrush) | *Inpainting* localizado cirúrgico | **13,97%** | **54 / 64** | 2 | 8 | **84,38%** |
| **Flux-Kontext** (AbstractEdit) | Edição contextual ampla (*Flow Matching*) | **48,24%** (até 99,5%) | **31 / 64** | **32** | 1 | **48,44%** |

### Explicação Conceitual e Física
1. **No *Inpainting* clássico (DALL-E 2):** A edição altera pequenas regiões focadas. O DinoLizer detecta uma área marcada pequena e concentrada. O classificador atinge **84,38% de recall** com quase zero confusão com imagens sintéticas.
2. **Na edição ampla (Flux-Kontext):** O modelo regenera o fundo inteiro, a iluminação ou elementos contextuais de larga escala, modificando em média metade da imagem (e em vários casos mais de 90%).
3. **Comportamento do Modelo:** Imagens integralmente sintéticas (SDXL, Midjourney) apresentam área marcada média de **66,84%**. Como o coeficiente da área marcada no Nível 2 é positivo ($+1{,}8695$), edições de grande extensão são empurradas probabilisticamente em direção à classe `synthetic`.
4. **Conclusão:** Não se trata de uma falha cega do pipeline, pois **98,44% das imagens do Flux-Kontext foram identificadas como IA** (apenas 1 foi classificada como real). Isso reflete uma fronteira conceitual legítima na literatura generativa: *em que ponto uma edição extensa por difusão deixa de ser uma edição pontual e se torna uma nova síntese de imagem?*

---

## 5. Histórico de Commits e Rastreabilidade

Todas as alterações foram estruturadas e integradas no Git seguindo o padrão de *Conventional Commits*:

1. `b149fc5` — `feat: implement joint Bayes decision rule in hierarchical classifier`
   - Implementação de $P(\text{real})$, $P(\text{synthetic})$ e $P(\text{edited})$ no runtime.
   - Adição do parâmetro `decision_rule="joint"`.
2. `2357d9f` — `fix(distildire): align default detector threshold with official ICML 2024 standard (0.5)`
   - Calibração do limiar nominal do DistilDIRE de acordo com o artigo original.
3. `5904fc4` — `refactor(classifier): remove legacy boolean policy and documentation`
   - Exclusão do classificador booleano, testes retrospectivos e artefatos de documentação `.md`.
4. `ccc201a` — `Merge branch 'experiment/joint-hierarchical-policy' into main`
   - Mesclagem limpa na branch principal e sincronização com o repositório remoto no GitHub.

---

## 6. Situação dos Arquivos de Texto do TCC

- **Repositório de Código:** Branch `main` perfeitamente sincronizada com o GitHub (`git push` concluído). Todos os 69 testes unitários aprovados.
- **Arquivo do Capítulo 4 da Monografia em LaTeX:** O arquivo em `/home/josue/Downloads/4-resultados-diffdetect-atualizado.tex` foi **preservado intacto em sua versão original**, aguardando as orientações e o momento desejado pelo aluno para a aplicação das atualizações de texto.

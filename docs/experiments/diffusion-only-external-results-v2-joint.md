# Resultados da Confirmação Externa — Agregação Hierárquica Conjunta de Bayes (Protocolo v2-Joint)

**Data de Execução:** 13 de novembro de 2026  
**Status dos Gates de Aceitação:** ✅ **100% APROVADO (6/6)**  
**Manifesto Avaliado:** [`diffusion-only-v2-manifest.csv`](results/diffusion-only-v2-manifest.csv) (`SHA-256: f3624b140bdcf486409953c74ed4d65bab25c78ff5c6a43d3287d5c7df242d33`)  
**Diretório de Artefatos:** [`docs/experiments/results/diffusion-only-v2-joint/`](results/diffusion-only-v2-joint/)

---

## 1. Visão Geral e Motivação Metodológica

A formulação anterior da política hierárquica realizava uma decisão em cascata com corte rígido (*hard thresholding*) no Nível 1:
- Se $P(\text{artificial}) < 0.27 \implies \text{Real}$;
- Se $P(\text{artificial}) \ge 0.73 \implies \text{Avança para o Nível 2}$;
- Se $0.27 \le P(\text{artificial}) < 0.73 \implies \text{Inconclusivo (Abstenção)}$.

Essa formulação apresentava duas limitações metodológicas:
1. **Presença de Decisões Inconclusivas:** Em sistemas práticos e avaliações científicas estritas, a existência de uma classe de abstenção reduz a utilidade operacional do sistema e mascara incertezas que devem ser penalizadas como erro.
2. **Propagação de Erro em Cascata (*Cascading Errors*):** Caso o Nível 1 cometa um erro com margem tênue (ex.: $P(\text{artificial}) = 0.52$), a imagem é irreversivelmente empurrada para o Nível 2, sendo forçada a ser rotulada como Sintética ou Editada, sem possibilidade de recuperação.

### A Solução: Regra da Cadeia de Probabilidade Hierárquica (Decisão Conjunta de Bayes)

Mantendo rigorosamente a estrutura hierárquica em dois níveis proposta no TCC 1, a regra de decisão conjunta calcula a probabilidade matemática total de cada classe folha por meio da regra da cadeia:

$$\begin{aligned}
P(\text{Real}) &= 1 - P(\text{Artificial}) \\
P(\text{Sintética}) &= P(\text{Artificial}) \times P(\text{Sintética} \mid \text{Artificial}) \\
P(\text{Editada}) &= P(\text{Artificial}) \times (1 - P(\text{Sintética} \mid \text{Artificial}))
\end{aligned}$$

A classificação final é atribuída à classe que maximiza a probabilidade a posteriori:
$$\hat{y} = \arg\max_{c \in \{\text{real}, \text{sintética}, \text{editada}\}} P(c \mid x)$$

**Vantagens Teóricas:**
* **Zero Inconclusivos:** 100% de cobertura de decisão sem zonas cinzentas artificiais.
* **Sem Limiares Arbitrários:** Elimina o limiar empírico $0.73$. A decisão decorre da Regra de Decisão Ótima de Bayes.
* **Mitigação da Soma de Erros:** Se o Nível 2 indicar forte incerteza sobre a natureza da manipulação, o ramo Real pode vencer a decisão conjunta caso o Nível 1 não tenha evidências robustas de artificialidade.

---

## 2. Comparativo de Desempenho

| Métrica | Hierárquica Cascata (Legado com Inconclusivo) | **Hierárquica Conjunta Bayes (v2-Joint)** | Meta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Acurácia Geral** | 76,04% | **78,91%** (303/384) | $\ge 70\%$ | ✅ Aprovado |
| **Acurácia Balanceada** | 76,04% | **78,91%** | $\ge 70\%$ | ✅ Aprovado |
| **Macro F1** | 77,94% | **78,75%** | $\ge 65\%$ | ✅ Aprovado |
| **Macro Precisão** | 80,03% | **78,66%** | — | Informativo |
| **Macro Recall** | 76,04% | **78,91%** | $\ge 50\%$ | ✅ Aprovado |
| **Cobertura de Decisão** | 95,31% | **100,00%** (384/384) | $\ge 80\%$ | ✅ Aprovado |
| **Taxa de Abstenção** | 4,69% | **0,00%** (0/384) | $\le 20\%$ | ✅ Aprovado |
| **Recall — Real** | 86,72% | **94,53%** (121/128) | $\ge 50\%$ | ✅ Aprovado |
| **Recall — Sintética** | 75,78% | **75,78%** (97/128) | $\ge 50\%$ | ✅ Aprovado |
| **Recall — Editada** | 65,62% | **66,41%** (85/128) | $\ge 50\%$ | ✅ Aprovado |

---

## 3. Matriz de Confusão Definitiva (Sem Inconclusivos)

```text
                  Predito Real   Predito Sintético   Predito Editado   Total
Real                   121               0                  7           128
Sintética (Difusão)      2              97                 29           128
Editada (Difusão)        9              34                 85           128
Total                  132             131                121           384
```

### Análise dos Fluxos de Decisão:
* **Classe Real (128 amostras):** 121 acertos (**94,53%**), 7 falsos positivos como editadas e 0 como sintéticas integrais. A precisão da classe Real alcançou **91,67%** com F1 de **93,08%**.
* **Classe Sintética (128 amostras):** 97 acertos (**75,78%**), 29 classificadas como editadas e apenas 2 falsos negativos como reais.
* **Classe Editada (128 amostras):** 85 acertos (**66,41%**), 34 confundidas como sintéticas e 9 como reais. A precisão alcançou **70,25%** e F1 de **68,27%**.

---

## 4. Desempenho por Gerador e Método

| Classe | Fonte / Gerador | Acertos | Taxa de Acerto |
| :--- | :--- | :---: | :---: |
| **Real** | Open Images V6 (validação) | 61 / 64 | **95,3%** |
| **Real** | RAISE-1k (fotografias RAW) | 60 / 64 | **93,8%** |
| **Sintética** | Midjourney v5 (*Synthbuster*) | 31 / 32 | **96,9%** |
| **Sintética** | Stable Diffusion XL (*Synthbuster*) | 21 / 32 | **65,6%** |
| **Sintética** | GenImage VQDM (*Discrete Diffusion*) | 23 / 32 | **71,9%** |
| **Sintética** | GenImage Wukong (*Latent Diffusion*) | 22 / 32 | **68,8%** |
| **Editada** | MagicBrush (DALL-E 2 Inpainting) | 54 / 64 | **84,4%** |
| **Editada** | AbstractEdit (Flux-Kontext) | 31 / 64 | **48,4%** |

---

## 5. Intervalos de Confiança por Bootstrap (2.000 Reamostragens)

| Métrica | Valor Observado | Média Bootstrap | Intervalo de Confiança 95% |
| :--- | :---: | :---: | :---: |
| **Acurácia** | 78,91% | 78,89% | **[74,74%; 82,81%]** |
| **Acurácia Balanceada** | 78,91% | 78,91% | **[75,04%; 82,58%]** |
| **Macro F1** | 78,75% | 78,68% | **[74,69%; 82,45%]** |
| **Cobertura** | 100,00% | 100,00% | **[100,00%; 100,00%]** |

---

## 6. Verificação dos Gates de Confirmação

| Critério de Aceitação | Meta | Observado | Status |
| :--- | :---: | :---: | :---: |
| `zero_aggregation_errors` | Erros = 0 | 0 erros | ✅ Aprovado |
| `coverage_at_least_80_percent` | $\ge 80\%$ | 100,00% | ✅ Aprovado |
| `balanced_accuracy_at_least_70_percent` | $\ge 70\%$ | 78,91% | ✅ Aprovado |
| `macro_f1_at_least_65_percent` | $\ge 65\%$ | 78,75% | ✅ Aprovado |
| `every_class_recall_at_least_50_percent` | Todos $\ge 50\%$ | Real: 94,5% \| Sint: 75,8% \| Edit: 66,4% | ✅ Aprovado |
| `all_confirmation_gates_passed` | Todos atendidos | 6 de 6 atendidos | ✅ Aprovado |

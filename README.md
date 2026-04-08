# Phi-4-mini-instruct Fine-tuning for Legal Case Outcome Classification

A thesis project comparing **LoRA** and **QLoRA** fine-tuning of [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) on a 7-class legal case outcome classification task.

---

## Overview

This repository contains Google Colab notebooks for fine-tuning and evaluating Phi-4-mini-instruct on the [`darklord1611/legal_citations`](https://huggingface.co/datasets/darklord1611/legal_citations) dataset. The goal is to predict the outcome of legal cases (`case_outcome`) from case text (`case_text`), and to compare the trade-offs between full LoRA (bfloat16) and quantised QLoRA (4-bit NF4) fine-tuning approaches.

Two experimental versions are provided:

- **v2** — baseline fine-tuning with a 5 000-sample class cap and a simple system prompt
- **v3** — improved fine-tuning with a tighter 1 500-sample class cap, `WeightedSFTTrainer` with inverse-frequency class weights, 2 training epochs, a lower learning rate, and a richer per-label system prompt

---

## Repository Structure

```
.
├── phi4_legal_base_evaluation.ipynb       # Zero-shot baseline (no fine-tuning)
│
├── phi4_legal_lora_v2.ipynb               # LoRA fine-tuning — v2 (5 000-cap)
├── phi4_legal_lora_evaluation_v2_.ipynb   # LoRA v2 evaluation
│
├── phi4_legal_lora_v3.ipynb               # LoRA fine-tuning — v3 (1 500-cap, weighted)
├── phi4_legal_Lora_evaluation_v3_.ipynb   # LoRA v3 evaluation
│
├── phi4_Qlora_legal_finetune_v2.ipynb     # QLoRA fine-tuning — v2 (5 000-cap)
├── phi4_Qlora_legal_evaluation_v2.ipynb   # QLoRA v2 evaluation
│
├── phi4_Qlora_legal_finetune_v3.ipynb     # QLoRA fine-tuning — v3 (1 500-cap, weighted)
└── phi4_Qlora_legal_evaluation_v3.ipynb   # QLoRA v3 evaluation
```

---

## Notebooks at a Glance

| Notebook | Purpose |
|---|---|
| `phi4_legal_base_evaluation` | Evaluates the unmodified Phi-4-mini-instruct (zero-shot) on both the 5 000-cap and 1 500-cap test splits as a baseline |
| `phi4_legal_lora_v2` | Fine-tunes with full LoRA (bfloat16, MAX_LABEL_COUNT = 5 000) |
| `phi4_legal_lora_evaluation_v2_` | Reports accuracy, precision, recall, F1, and confusion matrix for LoRA v2 |
| `phi4_legal_lora_v3` | Fine-tunes with full LoRA (bfloat16, MAX_LABEL_COUNT = 1 500, `WeightedSFTTrainer`, 2 epochs) |
| `phi4_legal_Lora_evaluation_v3_` | Reports metrics for LoRA v3 with the enriched system prompt |
| `phi4_Qlora_legal_finetune_v2` | Fine-tunes with QLoRA (4-bit NF4, MAX_LABEL_COUNT = 5 000) |
| `phi4_Qlora_legal_evaluation_v2` | Reports metrics for QLoRA v2 |
| `phi4_Qlora_legal_finetune_v3` | Fine-tunes with QLoRA (4-bit NF4, MAX_LABEL_COUNT = 1 500, `WeightedSFTTrainer`, 2 epochs) |
| `phi4_Qlora_legal_evaluation_v3` | Reports metrics for QLoRA v3 |

---

## Model & Dataset

| | |
|---|---|
| **Base model** | `microsoft/Phi-4-mini-instruct` (~3.8B parameters) |
| **Dataset** | `darklord1611/legal_citations` (HuggingFace Hub) |
| **Task** | 7-class legal case outcome classification |
| **Input column** | `case_text` |
| **Label column** | `case_outcome` |

---

## Key Hyperparameters

| Setting | v2 | v3 |
|---|---|---|
| `MAX_LABEL_COUNT` | 5 000 | 1 500 |
| `MIN_LABEL_COUNT` | 100 | 100 |
| LoRA rank `r` | 32 | 32 |
| LoRA alpha | 64 | 64 |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | `all-linear` | `all-linear` |
| Epochs | 1 | 2 |
| Learning rate | 2e-4 | 1e-4 |
| Per-device batch size | 2 | 2 |
| Gradient accumulation | 4 (effective batch = 8) | 4 (effective batch = 8) |
| Class weighting | None | Inverse-frequency (`WeightedSFTTrainer`) |
| Random seed | 42 | 42 |

### LoRA vs QLoRA

| Setting | QLoRA | LoRA |
|---|---|---|
| Quantisation | 4-bit NF4 (`BitsAndBytesConfig`) | None — full bfloat16 |
| VRAM requirement (est.) | ~12–16 GB | ~20–24 GB |
| Optimiser | `paged_adamw_8bit` | `adamw_torch` |

---

## Evaluation Metrics

All evaluation notebooks report:

- **Accuracy**
- **Precision** (macro & weighted)
- **Recall** (macro & weighted)
- **F1-score** (macro & weighted)
- **Per-class classification report**
- **Confusion matrix** (heatmap)

---

## Requirements

All notebooks are designed to run on **Google Colab** with a GPU runtime (A100 recommended for LoRA v3).

Key dependencies (installed automatically in each notebook):

```
torch (cu124)
transformers
peft
trl
accelerate
bitsandbytes
datasets
scikit-learn
pandas
matplotlib
seaborn
huggingface_hub
```

---

## Setup

1. Open any notebook in Google Colab.
2. Set your HuggingFace token where prompted (`HF_TOKEN = "hf_..."`).
3. Mount Google Drive — trained adapters and data splits are saved to `MyDrive`.
4. Run the dependency installation cell first, then **restart the runtime** before continuing.

> **Note:** Training notebooks save adapters and splits to Google Drive. Evaluation notebooks must point to the same Drive paths as the corresponding training notebook.

---

## Project Layout on Google Drive

```
MyDrive/
├── phi4_thesis_HuggingFace/        # QLoRA v2 outputs
│   ├── qlora_model/                # Adapter weights + tokenizer
│   ├── splits/                     # train / val / test CSVs + labels.json
│   └── metrics/
│
├── phi4_thesis_HuggingFace_v3/     # QLoRA v3 outputs
│   └── ...
│
├── phi4_thesis_LoRA_v2/            # LoRA v2 outputs
│   └── ...
│
├── phi4_thesis_LoRA_v3/            # LoRA v3 outputs
│   └── ...
│
└── phi4_thesis_BaseModel/          # Base model evaluation outputs
    ├── metrics_5000cap/
    └── metrics_1500cap/
```

---

## Citation

If you use this code in your research, please cite accordingly and reference:

- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
- Microsoft. (2024). [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct).
- Hu, E., et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).

---

## License

This repository is provided for academic and research purposes.

# 🫁 7-Day Postdoctoral Technical Challenge
### AI Medical Imaging · Visual Language Models · Semantic Retrieval

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-009EFF?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

> **Institution:** AlfaisalX · Cognitive Robotics and Autonomous Agents  
> **Unit:** MedX Research Unit, Medical Robotics & AI in Healthcare  
> **College:** Engineering and Advanced Computing, Alfaisal University, Riyadh  
> **Deadline:** February 22, 2026

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Task 1 — CNN Classification](#-task-1--cnn-classification)
- [Task 2 — Medical Report Generation](#-task-2--medical-report-generation)
- [Task 3 — Semantic Image Retrieval](#-task-3--semantic-image-retrieval)
- [Quick Start](#-quick-start)
- [Results Summary](#-results-summary)
- [Environment Setup](#-environment-setup)

---

## 🔍 Overview

This repository is a complete end-to-end AI pipeline for chest X-ray analysis combining three interconnected components:

| # | Task | Technology | Purpose |
|---|------|-----------|---------|
| 1 | **CNN Classification** | EfficientNet-B3 (Transfer Learning) | Pneumonia vs Normal detection |
| 2 | **Report Generation** | BLIP-2 / MedGemma (VLM) | Automated radiology report writing |
| 3 | **Image Retrieval** | CLIP + FAISS | Content-based similar-case search |

Built on the **Chest X-Ray Pneumonia** dataset (Kaggle), this system demonstrates how deep learning, large language models, and vector search can be combined into a clinical decision support prototype.

---

## 📂 Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

```
chest_xray/
├── train/
│   ├── NORMAL/       1,341 images
│   └── PNEUMONIA/    3,875 images
├── val/
│   ├── NORMAL/       8 images
│   └── PNEUMONIA/    8 images
└── test/
    ├── NORMAL/       234 images
    └── PNEUMONIA/    390 images
```

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

> ⚠️ **Class imbalance:** ~2.9:1 (PNEUMONIA:NORMAL) — handled via `WeightedRandomSampler`

---

## 📁 Repository Structure

```
7-Day-Postdoctoral-Technical-Challenge/
│
├── task1_classification/
│   └── task1_classification.py          # EfficientNet-B3 full pipeline
│
├── task2_report_generation/
│   └── task2_report_generation.py       # BLIP-2 / MedGemma VLM pipeline
│
├── task3_retrieval/
│   └── task3_retrieval.py               # CLIP + FAISS retrieval system
│
├── notebooks/
│   └── medical_imaging_challenge.ipynb  # Master Kaggle notebook (all 3 tasks)
│
├── reports/
│   ├── task1_classification_report.md
│   ├── task2_report_generation.md
│   └── task3_retrieval_system.md
│
├── requirements.txt
└── README.md
```

---

## 🧠 Task 1 — CNN Classification

**Objective:** Train a CNN to classify chest X-rays as Normal or Pneumonia, with thorough evaluation and failure case analysis.

### Model Architecture

```
EfficientNet-B3  (ImageNet pretrained)
│
├── Backbone ──── Frozen (except features.7 & features.8)
│
└── Classifier Head (trainable):
      Dropout(0.4)
   →  Linear(1536 → 256)
   →  SiLU
   →  Dropout(0.2)
   →  Linear(256 → 2)
```

**Why EfficientNet-B3?**  
Compound scaling of width, depth, and resolution achieves better accuracy per FLOP than ResNet at this scale. MBConv blocks with Squeeze-and-Excitation attention are well-suited for localising subtle radiological features. Pretrained ImageNet weights transfer low-level edge/texture detectors to the X-ray domain.

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 25 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | CosineAnnealingLR |
| Mixed precision | FP16 (`torch.cuda.amp`) |
| Imbalance strategy | `WeightedRandomSampler` |

### Augmentation

```
Resize(256) → RandomCrop(224) → RandomHorizontalFlip
→ RandomRotation(±10°) → ColorJitter → RandomAffine
→ Normalize (ImageNet μ/σ)
```

### Outputs

```
task1_outputs/
├── training_curves.png        — Loss, Accuracy, AUC, F1, Precision/Recall
├── confusion_matrix.png       — Raw + normalized confusion matrices
├── roc_curve.png              — ROC with Youden's J optimal threshold
├── sample_predictions.png     — 16 random test predictions
├── failure_cases.png          — Misclassified images with confidence
└── task1_classification_report.md

models/
└── best_efficientnet_b3.pth   — Best checkpoint (saved by Val AUC)
```

### Run

```bash
python task1_classification/Task_1_code.py
```

---

## 📝 Task 2 — Medical Report Generation

**Objective:** Use a Visual Language Model (VLM) to automatically generate natural language radiology reports from chest X-ray images.

### Model Options

| Model | Default? | Notes |
|-------|----------|-------|
| **BLIP-2 OPT-2.7B** | ✅ Yes | No auth required, free GPU tier |
| **MedGemma-4B-IT** | ⭐ Preferred | Medical-domain trained, needs HF token |

MedGemma is recommended for production as it is pre-trained on radiology data and produces more accurate clinical terminology. To enable it:

```bash
# 1. Accept license: https://huggingface.co/google/medgemma-4b-it
# 2. Set environment variable — NEVER commit tokens to Git
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxx
# In Kaggle: Notebook Settings → Secrets → Add Secret → Key: HUGGINGFACE_TOKEN
```

### Prompting Strategies Tested

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `basic` | "Describe the findings in this chest X-ray." | Baseline |
| `clinical_structured` | Lung Fields → Cardiac → Pleural → Impression | Best clinical output |
| `differential` | Radiological features + differential diagnosis | Ambiguous cases |
| `clinical_brief` | Binary verdict + key finding, concise | Triage |

### Pipeline

```
Chest X-Ray (upscaled to 512×512 if small)
      │
      ▼
   VLM Encoder + Structured Prompt
      │
      ▼
   Generated Report Text
      │
      ▼
   Keyword Alignment Scoring vs Ground Truth
```

### Outputs

```
task2_outputs/
├── report_cards.png           — Image + report side-by-side (6 cards)
├── prompt_comparison.png      — 4 strategies compared on same image
├── all_reports.csv            — 40 reports (10 images × 4 prompts)
└── task2_report_generation.md
```

### Run

```bash
python task2_report_generation/Task_2_code.py
```

---

## 🔍 Task 3 — Semantic Image Retrieval

**Objective:** Build a Content-Based Image Retrieval (CBIR) system so clinicians can find visually similar X-ray cases from a database.

### System Architecture

```
Query: Image  ──┐
Query: Text   ──┴──▶  CLIP / BiomedCLIP Encoder
                           │  512-dim L2-normalized vector
                           ▼
                     FAISS IndexFlatIP
                           │  cosine similarity search
                           ▼
                      Top-k Results
                 (path · label · similarity score)
```

### Embedding Model Comparison

| Model | Dim | Text Search | Medical Domain |
|-------|-----|-------------|----------------|
| **CLIP ViT-B/32** *(default)* | 512 | ✅ | ❌ General |
| **BiomedCLIP** *(preferred)* | 512 | ✅ | ✅ 15M PubMed pairs |

### CLI Usage

```bash
# Build index + full evaluation
python task3_retrieval/Task_3_code.py --mode full

# Image-to-image search
python task3_retrieval/task3_retrieval.py \
    --mode search_image \
    --query /path/to/xray.jpeg \
    --k 5

# Text-to-image search
python task3_retrieval/task3_retrieval.py \
    --mode search_text \
    --query "bilateral consolidation with air bronchograms" \
    --k 5
```

### Evaluation — Precision@k

> **P@k** = (# top-k results sharing query label) / k  
> Random binary baseline = **0.500**  
> Values above 0.5 confirm embeddings meaningfully cluster similar pathology.

### Outputs

```
task3_outputs/
├── retrieval_results.png      — Query | Top-5 grid (green=match, red=mismatch)
├── text_retrieval_results.png — Clinical text queries → retrieved images
├── precision_at_k.png         — P@{1,3,5,10} bar chart vs baseline
├── tsne_embeddings.png        — t-SNE 2D of CLIP embedding space
└── task3_retrieval_system.md

task3_outputs/index/
├── test_embeddings.npy        — 624 × 512 float32 embedding matrix
├── test_metadata.json         — Image paths and labels
└── test_index.faiss           — FAISS IndexFlatIP
```

---

## 🚀 Quick Start

### Option A — Kaggle (Recommended)

```
1. Create a new Kaggle notebook
2. Add dataset: chest-xray-pneumonia
3. Upload: notebooks/medical_imaging_challenge.ipynb
4. Accelerator: GPU T4 x2
5. Run All
```

### Option B — Local

```bash
# Clone
git clone https://github.com/Maisamilens/7-Day-Postdoctoral-Technical-Challenge.git
cd 7-Day-Postdoctoral-Technical-Challenge

# Install
pip install -r requirements.txt

# Update DATA_ROOT in each script to your local dataset path, then:
python task1_classification/task1_classification.py
python task2_report_generation/task2_report_generation.py
python task3_retrieval/task3_retrieval.py --mode full
```

---

## 📊 Results Summary

### Task 1 — Test Set Performance (EfficientNet-B3)

| Metric | Score |
|--------|-------|
| Accuracy | ~93% |
| Precision | ~93% |
| **Recall (Sensitivity)** | **~96%** |
| Specificity | ~88% |
| F1-Score | ~94% |
| **AUC-ROC** | **~97%** |

> Recall is the most clinically critical metric — a missed pneumonia (false negative) is more dangerous than a false positive.

### Task 3 — Retrieval Precision@k (CLIP ViT-B/32)

| k | P@k | vs Baseline |
|---|-----|------------|
| 1 | ~0.78 | +0.28 |
| 3 | ~0.76 | +0.26 |
| 5 | ~0.75 | +0.25 |
| 10 | ~0.73 | +0.23 |

> Baseline (random binary retrieval) = 0.500

---

## ⚙️ Environment Setup

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.41.0
accelerate>=0.27.0
sentencepiece
faiss-cpu
open_clip_torch
scikit-learn>=1.3.0
seaborn>=0.13.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.65.0
nbformat>=5.9.0
```

### Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Task 1 — Training | CPU (slow) | Kaggle T4 GPU |
| Task 2 — VLM Inference | T4 16GB | A100 (for MedGemma) |
| Task 3 — Embeddings + FAISS | CPU | T4 GPU |

---

## 🔬 Design Decisions

**Class Imbalance (2.9:1)** — Handled using `WeightedRandomSampler`. The minority class (NORMAL) is sampled with proportionally higher probability, avoiding image duplication which would cause overfitting.

**FAISS Index** — `IndexFlatIP` (exact cosine similarity on L2-normalized vectors) was chosen over approximate methods because the 624-vector test set is small enough for exact search without speed compromise.

**VLM Upscaling** — Images smaller than 256px are upscaled to 512×512 (bicubic) before VLM inference to improve attention-layer feature extraction.

**Failure Cases** — Common misclassification patterns: (1) mild pneumonia with subtle consolidation resembles normal at 224×224; (2) pulmonary edema occasionally misclassified as pneumonia; (3) non-standard projections deviate from standard PA-view training distribution.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with PyTorch · HuggingFace Transformers · FAISS · OpenAI CLIP</p>

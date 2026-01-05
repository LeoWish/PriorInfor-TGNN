
# PriorInfor-TGNN

**Prior-Informed Transformer Graph Neural Network for Predicting Synergistic Multi-Strain Starter Cultures**

This repository contains the official implementation of **PriorInfor-TGNN**, a prior-informed Transformer-based Graph Neural Network proposed in the manuscript:

> **PriorInfor-TGNN Enables Predictive and Mechanistic Design of Synergistic Multi-Strain Dairy Starter Cultures**

The framework integrates **biological prior knowledge**, **graph neural networks**, and **Transformer-based attention** to predict synergistic interactions in multi-strain dairy starter cultures.

---

## 📁 Repository Structure

```text
PriorInfor-TGNN/
├── LapRLS.py                # Laplacian Regularized Least Squares (prior inference)
├── TGNN-5-fold-CV.py        # PriorInfor-TGNN with 5-fold cross-validation
├── ML-5-fold-CV.py          # Baseline machine learning models with 5-fold CV
├── data/
│   ├── dataset.xlsx           # Main dataset (features and labels)
│   ├── LapRLS_Pre.xlsx        # LapRLS predictions and gold-standard interactions
│   └── similarity_matrix.xlsx # Similarity matrices for LapRLS
└── README.md

```

---

## 🔬 Workflow Overview

The complete pipeline consists of **three sequential stages**:

1. **Prior Interaction Inference (LapRLS)** Infer potential interactions between *Lactobacillus delbrueckii subsp. bulgaricus* (Lb) and *Streptococcus thermophilus* (St) strains using similarity-based learning.
2. **Graph-Based Deep Learning (PriorInfor-TGNN)** Integrate inferred interaction priors and genomic features into a Transformer-based Graph Neural Network.
3. **Baseline Comparison (Conventional Machine Learning)** Benchmark PriorInfor-TGNN against multiple classical ML models using identical 5-fold cross-validation.

---

## 🧠 Script Descriptions

### 1. `LapRLS.py` — Prior Interaction Inference

**Purpose:** Implements a **Laplacian Regularized Least Squares (LapRLS)** model to infer interaction scores between Lb–St strain pairs based on multiple similarity matrices.

* **Main Steps:** Linear fusion of similarity matrices, symmetric normalization, and Laplacian regularization-based prediction.
* **Input:** `data/similarity_matrix.xlsx` (Lb/St similarity matrices and known interaction adjacency matrix).
* **Output:** `LapRLS_pre.csv` (Predicted interaction scores used as **soft prior edges** in the TGNN).

**Run:**

```bash
python LapRLS.py

```

### 2. `TGNN-5-fold-CV.py` — PriorInfor-TGNN

**Purpose:** Implements the Prior-Informed Transformer Graph Neural Network, integrating node features and edge-level prior knowledge.

* **Model Characteristics:** * Transformer-based graph convolutions (`TransformerConv`).
* Edge attributes: `interaction_score` and `is_gold_standard`.
* Class imbalance handled via weighted binary cross-entropy loss.


* **Input:** `data/dataset.xlsx` (Features/Labels) and `data/LapRLS_Pre.xlsx` (Interaction priors).
* **Metrics:** ROC-AUC, PR-AUC, and Pooled out-of-fold performance.

**Run:**

```bash
python TGNN-5-fold-CV.py

```

### 3. `ML-5-fold-CV.py` — Baseline Machine Learning

**Purpose:** Provides a systematic comparison between PriorInfor-TGNN and classical ML approaches.

* **Included Models:** Logistic Regression, SVM, Random Forest, KNN, Naive Bayes, Gradient Boosting, XGBoost, LightGBM, CatBoost, and MLP.
* **Output:** `models_performance_summary.csv` (Mean ± SD AUROC/AUPR).

**Run:**

```bash
python ML-5-fold-CV.py

```

---

## 📦 Data Description

All datasets are stored in the `data/` directory:

| File | Description |
| --- | --- |
| `dataset.xlsx` | Main dataset containing strain combinations, labels, and genomic features. |
| `similarity_matrix.xlsx` | Multi-source similarity matrices used for LapRLS. |
| `LapRLS_Pre.xlsx` | LapRLS predictions and gold-standard interaction pairs. |

---

## ⚙️ Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.12
* PyTorch Geometric
* scikit-learn
* pandas, numpy, scipy
* xgboost, lightgbm, catboost

*GPU acceleration is optional but recommended for TGNN training.*

---

## 🔁 Reproducibility

* Fixed random seeds are used across all scripts.
* Stratified 5-fold cross-validation ensures fair evaluation.
* Pooled out-of-fold predictions are used for final performance estimation.

```

```

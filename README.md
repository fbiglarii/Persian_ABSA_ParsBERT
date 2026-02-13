# Persian ABSA: Fine-tuning ParsBERT for Aspect-Based Sentiment Analysis

> **Course Project:** Fine-tuning ParsBERT for ABSA on Persian restaurant reviews. Achieved **90%+ accuracy** with only 240 training samples.

---

## 🎯 Overview

**Task:** Given Persian text and an aspect, classify sentiment as `positive`, `negative`, or `neutral`.

**Example:**
```
Text: "غذای این رستوران عالی بود ولی سرویس کند بود"
Aspect: "غذا" → Sentiment: positive ✅
```

---

## 📖 Background

> **See [`EXPERIMENTS.docx`](EXPERIMENTS.docx) for baseline experiments and motivation.**

**Previous Experiments:**
- InstructABSA (English model) on Persian: **40% accuracy** ❌
- **Problem:** English models insufficient for Persian

**Current Approach:**
- Fine-tuned ParsBERT (Persian model): **90%+ accuracy** ✅
- **Improvement:** +50% over baseline

---

## 📊 Dataset

- **Source:** Persian restaurant reviews
- **Size:** ~280 samples
  - Train: 240 (80%)
  - Test: 60 (20%)
- **Classes:** positive, negative, neutral
- **Format:** CSV (`persian_train.csv`, `persian_test.csv`)

---

## 🛠️ Method

**Input Format:** `[text] [SEP] [aspect]`

**Model:**
- Base: ParsBERT (`HooshvareLab/bert-fa-base-uncased`)
- Architecture: BERT-base (12 layers, 768 hidden, ~110M params)
- Task: 3-class classification

**Training:**
- Epochs: 10
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW
- Time: ~15 minutes (T4 GPU)

---

## 📈 Results

| Model | Accuracy | Description |
|-------|----------|-------------|
| Random Baseline | 33.33% | Random guessing |
| InstructABSA (English) | ~40% | English instruction-tuned model |
| **ParsBERT (Fine-tuned)** | **90%+** ✅ | Our approach |

**Key Finding:** Persian pre-trained model essential for Persian ABSA (+50% improvement).

---

## 🚀 Quick Start

### Google Colab (Recommended)

1. Open `Persian_ABSA_ParsBERT.ipynb` in Colab
2. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`
3. Upload `persian_train.csv` and `persian_test.csv`
4. Run all cells: `Runtime` → `Run all`

**Runtime:** ~15-20 minutes

### Local Setup

```bash
pip install transformers datasets accelerate scikit-learn pandas numpy
jupyter notebook Persian_ABSA_ParsBERT.ipynb
```

---

## ⚠️ Limitations

1. **Small dataset** (~280 samples): Results may vary by split; k-fold CV recommended
2. **Single split reporting:** One fixed split may not generalize
3. **Domain-specific:** Trained only on restaurant reviews
4. **Validation strategy:** Test set used for monitoring (potential leakage if hyperparameters tuned on test)

---


## 📁 Files

```
├── README.md                          # This file
├── EXPERIMENTS.docx                   # Baseline experiments
├── Persian_ABSA_ParsBERT.ipynb        # Training notebook
├── persian_train.csv                  # Training data
└── persian_test.csv                   # Test data
```

---

## 📚 References

**Papers:**
1. Devlin et al., 2019 - "BERT: Pre-training of Deep Bidirectional Transformers"
2. Farahani et al., 2020 - "ParsBERT: Transformer-based Model for Persian Language"
3. Scaria et al., 2024 - "Instruction Learning for ABSA"

**Resources:**
- [ParsBERT Model](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 🙏 Acknowledgments

- **HooshvareLab** for ParsBERT
- **Google Colab** for free GPU
- **Course Instructors** for guidance



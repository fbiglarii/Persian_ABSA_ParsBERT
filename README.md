# Persian ABSA: Fine-tuning ParsBERT for Aspect-Based Sentiment Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 🎯 Project Overview

This project implements **Aspect-Based Sentiment Analysis (ABSA)** for Persian language by fine-tuning the **ParsBERT** model, achieving **90%+ accuracy** on Persian restaurant reviews.

### Key Results
- ✅ **Accuracy: 90%+** (ParsBERT fine-tuned)
- 📊 **+50% improvement** over English baseline
- 🚀 Training time: **~15 minutes** on Google Colab T4 GPU

---

## 📖 Background: Previous Experiments

> **Important:** Before diving into this project, please read [`EXPERIMENTS.docx`](EXPERIMENTS.docx) to understand our journey and baseline results.


### Our Project:

**Solution:** Fine-tune a Persian pre-trained model (ParsBERT) specifically for ABSA

**Result:** 
- ParsBERT (fine-tuned): **90%+ accuracy** ✅
- **2.25x better** than English baseline
- **+50% improvement** overall

---

## 📊 Dataset

- **Source:** Persian restaurant reviews
- **Total samples:** ~280
  - Training: 240 samples (80%)
  - Test: 60 samples (20%)
- **Classes:** positive, negative, neutral
- **Format:** CSV files (`persian_train.csv`, `persian_test.csv`)

**Files included:**
- `persian_train.csv` - Training data
- `persian_test.csv` - Test data

---

## 🚀 Quick Start

### Prerequisites
- Google Account (for Colab)
- GPU runtime (T4 recommended)

### Step-by-Step Guide

#### 1. Open Notebook in Google Colab
1. Upload `Persian_ABSA_ParsBERT.ipynb` to Google Colab
2. Enable GPU: `Runtime` → `Change runtime type` → Select `GPU (T4)`

#### 2. Upload Data Files
Upload both CSV files to Colab:
- `persian_train.csv`
- `persian_test.csv`

#### 3. Run the Notebook
Execute all cells: `Runtime` → `Run all`

**⏱️ Total runtime:** ~15-20 minutes

---

## 🧠 Understanding Fine-tuning

### Common Question: Does fine-tuning change the model?

**Answer:** No! The base model architecture remains the same, but it becomes **specialized** for ABSA.

### Before Fine-tuning:
```
ParsBERT (Pre-trained)
│
├─ Capability: Understanding Persian language
├─ Task: Language Modeling (MLM)
├─ Training: Millions of Persian sentences
└─ Output: Word embeddings (768-dim vectors)
```
❌ **Cannot** classify sentiments

### After Fine-tuning:
```
ParsBERT (Fine-tuned for ABSA)
│
├─ Capability: Understanding Persian + Sentiment detection
├─ Task: Aspect-Based Sentiment Classification
├─ Training: +240 labeled ABSA samples
└─ Output: [P(positive), P(negative), P(neutral)]
```
✅ **Can** classify sentiments accurately!

### What Changed?

| Aspect | Before | After |
|--------|--------|-------|
| **Architecture** | BERT-base (12 layers) | Same ✓ |
| **Tokenizer** | ParsBERT tokenizer | Same ✓ |
| **Weights** | General Persian | Specialized for ABSA ✓ |
| **Output Layer** | Embeddings only | + Classification head (3 classes) ✓ |
| **Capability** | Language understanding | Understanding + Sentiment detection ✓ |

**Summary:** Same architecture, slightly updated weights, new capability!

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Description |
|-------|----------|-------------|
| Random Baseline | 33.33% | Random guessing |
| InstructABSA (English) | ~40% | English instruction-tuned model |
| **ParsBERT (Fine-tuned)** | **90%+** ✅ | Our approach |

### Key Findings

1. **Persian models are essential:** 
   - English models: 40%
   - Persian models: 90%+
   - **Improvement: +50%**

2. **Fine-tuning is effective:**
   - With only 240 training samples
   - Achieved 90%+ accuracy
   - Training time: ~15 minutes

3. **Transfer learning works:**
   - Leveraged ParsBERT's Persian knowledge
   - Adapted for ABSA task
   - No need to train from scratch

---

## 🛠️ Technical Details

### Model Configuration
- **Base Model:** ParsBERT (`HooshvareLab/bert-fa-base-uncased`)
- **Architecture:** BERT-base
  - 12 transformer layers
  - 768 hidden dimensions
  - 12 attention heads
  - ~110M parameters

### Training Hyperparameters
- **Epochs:** 10
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Warmup Steps:** 50
- **Weight Decay:** 0.01
- **Optimizer:** AdamW

### Why These Settings?

#### Why 10 Epochs?
- Small dataset (240 samples) requires more iterations
- Fine-tuning (not training from scratch) needs 5-15 epochs
- Standard BERT fine-tuning practice
- Achieved optimal results without overfitting

#### Why Batch Size 16?
- Balance between speed, memory, and quality
- 240 samples ÷ 16 = 15 steps per epoch
- Optimal for T4 GPU memory
- Standard for BERT fine-tuning

#### Train/Test Split: 80/20
- Training: 240 samples (larger for better learning)
- Test: 60 samples (evaluation)
- No separate validation set due to small dataset size
- Used early stopping to prevent overfitting

---

## 💡 Key Insights

### 1. Why ParsBERT?
- ✅ Pre-trained on Persian text
- ✅ Better understanding of Persian grammar
- ✅ Superior to English models (2.25x better)

### 2. Why Fine-tuning?
- ✅ Leverages existing knowledge
- ✅ Requires less training data
- ✅ Faster training
- ✅ Better results than training from scratch

### 3. Why This Approach Succeeded?
- ✅ Strong base model (ParsBERT)
- ✅ Quality dataset (Persian reviews)
- ✅ Proper hyperparameter tuning
- ✅ Transfer learning benefits

---

## 📁 Project Structure

```
persian-absa-finetuning/
├── README.md                          # This file
├── EXPERIMENTS.docx                   # experiments & baseline
├── Persian_ABSA_ParsBERT.ipynb       # Main training notebook
├── persian_train.csv                  # Training data
└── persian_test.csv                   # Test data
```

---

## 🔬 Methodology

### 1. Data Preparation
- Load CSV files with Persian reviews
- Each row contains: text, aspect, sentiment
- Combine text and aspect with `[SEP]` token
- Convert sentiment labels to numeric (0, 1, 2)

### 2. Model Loading
- Load pre-trained ParsBERT from HuggingFace
- Add classification head (3 output classes)
- Initialize with pre-trained weights

### 3. Training
- Fine-tune for 10 epochs
- Monitor validation accuracy each epoch
- Save best model based on accuracy
- Use early stopping if needed

### 4. Evaluation
- Test on held-out test set (60 samples)
- Calculate accuracy, precision, recall, F1
- Generate confusion matrix
- Analyze per-class performance

---

## 📚 References

### Papers
1. **BERT:** Devlin et al., 2019 - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. **ParsBERT:** Farahani et al., 2020 - "ParsBERT: Transformer-based Model for Persian Language Understanding"
3. **InstructABSA:** Scaria et al., 2024 - "Instruction Learning for Aspect-Based Sentiment Analysis"

### Resources
- [ParsBERT GitHub](https://github.com/hooshvare/parsbert)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [ParsBERT Model Card](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)





# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Links

### W&B Report
https://api.wandb.ai/links/ce23b108-indian-institute-of-technology-madras/5zb4juxs

### GitHub Repository
https://github.com/sagar-250/da6401_assignment_3.git

---

## Overview

This assignment implements the landmark Transformer architecture from the paper **"Attention Is All You Need"** using PyTorch. The objective is to build a Neural Machine Translation (NMT) system capable of translating German sentences into English using the Multi30k dataset.

The implementation includes:
- Multi-Head Self Attention
- Positional Encoding
- Encoder-Decoder Transformer Architecture
- Noam Learning Rate Scheduler
- Label Smoothing
- Attention Visualization and Analysis
- BLEU Score Evaluation

---

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and tokenization
├── train.py           # Training and inference pipeline
```

---

## Features Implemented

- Transformer Encoder-Decoder architecture from scratch
- Scaled Dot Product Attention
- Multi-Head Attention
- Sinusoidal Positional Encoding
- Learned Positional Embeddings
- Noam Learning Rate Scheduler
- Label Smoothing
- Attention Head Visualization
- BLEU Score Evaluation
- W&B Experiment Tracking

---

## Experiments Performed

### 2.1 Noam Scheduler vs Fixed Learning Rate
- Compared Transformer training stability using:
  - Noam Scheduler
  - Fixed learning rate

### 2.2 Scaling Factor Ablation
- Compared attention with and without:
  - `1 / sqrt(d_k)`

### 2.3 Attention Rollout & Head Specialization
- Visualized attention maps for all encoder heads
- Analyzed:
  - head specialization
  - long-range dependencies
  - head redundancy

### 2.4 Positional Encoding vs Learned Embeddings
- Compared:
  - sinusoidal positional encoding
  - learned positional embeddings

### 2.5 Label Smoothing
- Compared:
  - epsilon_ls = 0.0
  - epsilon_ls = 0.1

---

- PyTorch Documentation
- Multi30k Dataset
- Weights & Biases

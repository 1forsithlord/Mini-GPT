# Mini-GPT (Shakespeare)

A minimal, from-scratch implementation of a GPT (Generative Pre-trained Transformer) language model, trained on the Tiny Shakespeare dataset. This repository is designed to be an educational tool to understand the core mechanics of LLMs, including Causal Self-Attention, Multi-Layer Perceptrons, and autoregressive text generation.

## Features
- **From Scratch**: No Hugging Face `transformers` library. Pure PyTorch.
- **Hardware Agnostic**: Runs on Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU.
- **Regularization**: Includes Dropout and Cosine Annealing Learning Rate scheduling to prevent overfitting.

## Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy tqdm
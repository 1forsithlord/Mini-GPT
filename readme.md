# Mini-GPT: Building Shakespeare from Scratch 🎭

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Ever wondered what actually happens under the hood of ChatGPT? I did. So I decided to build a tiny version of a GPT (Generative Pre-trained Transformer) from absolute scratch using only raw PyTorch. 

No Hugging Face wrappers, no magic black boxes—just pure, readable code. I trained it on the Tiny Shakespeare dataset, and it learned to write its own (very dramatic) Elizabethan plays.

## How it works (The Flow), BOOM BOOM

Here’s a quick visual of how the data moves through the project:

```mermaid
graph TD
    A["input.txt"] -->|"prepare_data.py"| B["train.bin / val.bin"]
    B -->|"train.py"| C["get_batch"]
    
    subgraph Model["Model.py (The Brain)"]
        C --> D["Token & Position Embeddings"]
        D --> E["Transformer Block 1"]
        E --> F["..."]
        F --> G["Transformer Block N"]
        G --> H["LayerNorm"]
        H --> I["Linear LM Head"]
    end
    
    I --> J["Logits"]
    J -->|"Cross Entropy Loss"| K["Backpropagation"]
    K -->|"AdamW + Cosine Decay"| E
    
    I -->|"sample.py"| L["Generated Shakespeare Text"]

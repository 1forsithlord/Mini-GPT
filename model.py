import os
import pickle
import torch
from model import GPT


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6


# 1. Load the vocabulary mapping
data_dir = os.path.join(os.path.dirname(__file__), 'data')
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 2.LETs Re-initialize the model 
model = GPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=0.0)

# 3.Okie Load our trained weights!
model.load_state_dict(torch.load('model.pt', map_location=device))
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}!")

# 4. The Generation Function
@torch.no_grad()
def generate(prompt, max_new_tokens=250, temperature=0.7):
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return decode(idx[0].tolist())

# 5. prompts to generate text from
prompts = [
    "JULIET: O Romeo, Romeo! wherefore art thou",
    "To be, or not to be, that is the",
    "KING HENRY: I have seen too much"
]

for p in prompts:
    print(f"\n--- Prompt: '{p}' ---")
    result = generate(p, max_new_tokens=250, temperature=0.7)
    print(result)
    print("-" * 40)
    
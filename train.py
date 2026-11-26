import os
import time
import math
import numpy as np
import torch

from model import GPT

# -----------------------------------------------------------------------------
# Hyperparameters
batch_size = 64
block_size = 256       # Max context length for predictions
max_iters = 3000       # Total training steps
eval_interval = 300    # How often to evaluate
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2 
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"Using device: {device}")
torch.manual_seed(1337)

# 1. Load the data
data_dir = 'data'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Determine vocab size
vocab_size = int(max(train_data.max(), val_data.max())) + 1

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 2. Initialize the model
model = GPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=dropout)
m = model.to(device)

print(f"Model initialized with {sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

# 3. Setup Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

# 4. Helper function to estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 5. The Training Loop
print("Starting training...")
start_time = time.time()

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

end_time = time.time()
print(f"Training finished in {(end_time - start_time)/60:.2f} minutes")

# 6. Save the model
torch.save(model.state_dict(), 'model.pt')
print("Model saved to model.pt")
import os
import urllib.request
import pickle
import numpy as np

# 1. Download the Tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

file_path = os.path.join(data_dir, 'input.txt')
if not os.path.exists(file_path):
    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(url, file_path)

# 2. Read the text
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset length: {len(text):,} characters")

# 3. Get all unique characters in the text (our vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

# 4. Create mappings from character to integer, and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Helper functions to encode/decode
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 5. Split into Train and Validation sets (90% train, 10% val)
n = len(text)
train_text = text[:int(n*0.9)]
val_text = text[int(n*0.9):]

# Encode the text into integers
train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)

# 6. Save the binary files for training
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# 7. Save the vocabulary mapping so we can decode generated text later
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)

print("Data prepared! Files saved to the 'data' folder.")
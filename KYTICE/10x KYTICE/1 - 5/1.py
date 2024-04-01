```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # Velikost dávky určuje, kolik nezávislých sekvencí bude zpracováno paralelně.
block_size = 256 # Maximální délka kontextu pro predikce.
max_iters = 5000 # Maximální počet iterací trénování.
eval_interval = 500 # Interval, po kterém se provede evaluace modelu.
learning_rate = 3e-4 # Rychlost učení určuje, jak rychle se model učí z dat.
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Použití GPU pokud je dostupné, jinak CPU.
eval_iters = 200 # Počet iterací pro evaluaci.
n_embd = 384 # Velikost vektoru vložení.
n_head = 6 # Počet hlav v multi-head attention mechanismu.
n_layer = 6 # Počet vrstev v modelu.
dropout = 0.2 # Pravděpodobnost vynechání neuronu během trénování.
# ------------

torch.manual_seed(1337) # Nastavení seedu pro reprodukovatelnost výsledků.

# wget https://raw.githubusercontent.com/koliby777/pokus-cislo/master/kytice.txt
with open('10x kytice.txt', 'r', encoding='utf-8') as f:
    text = f.read() # Načtení textu z souboru.

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # Seznam všech unikátních znaků v textu.
vocab_size = len(chars) # Velikost slovníku.
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # Mapování znaků na celá čísla.
itos = { i:ch for i,ch in enumerate(chars) } # Mapování celých čísel na znaky.
encode = lambda s: [stoi[c] for c in s] # Funkce pro kódování řetězce na seznam celých čísel.
decode = lambda l: ''.join([itos[i] for i in l]) # Funkce pro dekódování seznamu celých čísel na řetězec.

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # Kódování celého textu na tensor.
n = int(0.9*len(data)) # Prvních 90% dat pro trénink, zbytek pro validaci.
train_data = data[:n] # Tréninková data.
val_data = data[n:] # Validační data.

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # Rozdělení dat na tréninková a validační.
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Náhodný výběr začátků sekvencí.
    x = torch.stack([data[i:i+block_size] for i in ix]) # Vstupní data pro model.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Cílová data pro model.
    x, y = x.to(device), y.to(device) # Přesun dat na zařízení (GPU/CPU).
    return x, y
```
```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # určuje, kolik nezávislých sekvencí bude zpracováváno paralelně
block_size = 256 # maximální délka kontextu pro predikce
max_iters = 5000 # maximální počet iterací trénování
eval_interval = 500 # interval pro evaluaci modelu
learning_rate = 3e-4 # rychlost učení
device = 'cuda' if torch.cuda.is_available() else 'cpu' # zařízení pro výpočty, GPU pokud je dostupné, jinak CPU
eval_iters = 200 # počet iterací pro evaluaci
n_embd = 384 # velikost vektorů vložení
n_head = 6 # počet hlav v multi-head attention
n_layer = 6 # počet vrstev transformátoru
dropout = 0.2 # pravděpodobnost dropoutu
# ------------

torch.manual_seed(1337) # nastaví náhodný seed pro reprodukovatelnost

# wget https://raw.githubusercontent.com/koliby777/pokus-cislo/master/kytice.txt
with open('10x kytice.txt', 'r', encoding='utf-8') as f:
    text = f.read() # načte textový soubor

# zde jsou všechny unikátní znaky, které se v textu vyskytují
chars = sorted(list(set(text))) # vytvoří seznam unikátních znaků
vocab_size = len(chars) # počet unikátních znaků
# vytvoří mapování znaků na celá čísla
stoi = { ch:i for i,ch in enumerate(chars) } # mapování znak na index
itos = { i:ch for i,ch in enumerate(chars) } # mapování index na znak
encode = lambda s: [stoi[c] for c in s] # encoder: převede řetězec na seznam čísel
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: převede seznam čísel zpět na řetězec

# Rozdělení dat na trénovací a validační
data = torch.tensor(encode(text), dtype=torch.long) # zakóduje celý text do tensoru
n = int(0.9*len(data)) # prvních 90% dat bude trénovacích, zbytek validačních
train_data = data[:n] # trénovací data
val_data = data[n:] # validační data

# načítání dat
def get_batch(split):
    # vygeneruje malý batch dat pro vstupy x a cíle y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # náhodně vybere začátky sekvencí
    x = torch.stack([data[i:i+block_size] for i in ix]) # vytvoří vstupy x
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # vytvoří cíle y (o jednu pozici posunuté)
    x, y = x.to(device), y.to(device) # přesune data na zvolené zařízení
    return x, y
```
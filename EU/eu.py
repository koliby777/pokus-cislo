# 1 ---------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # určuje, kolik nezávislých sekvencí bude zpracováváno paralelně
block_size = 32 # maximální délka kontextu pro predikce
max_iters = 500 # maximální počet iterací trénování
eval_interval = 100 # interval pro evaluaci modelu
learning_rate = 3e-3 # rychlost učení
device = 'cuda' if torch.cuda.is_available() else 'cpu' # zařízení pro výpočty, GPU pokud je dostupné, jinak CPU
eval_iters = 200 # počet iterací pro evaluaci
n_embd = 384 # velikost vektorů vložení
n_head = 6 # počet hlav v multi-head attention
n_layer = 6 # počet vrstev transformeru
dropout = 0.0 # pravděpodobnost dropoutu


torch.manual_seed(1337) # nastaví náhodný seed pro reprodukovatelnost

# !wget https://raw.githubusercontent............
with open('eu.txt', 'r', encoding='utf-8') as f:
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

# 2 ---------------------------------------------------------------------------------

@torch.no_grad() # deaktivuje výpočet gradientů pro zvýšení efektivity
def estimate_loss():
    out = {}
    model.eval() # přepne model do evaluačního módu
    for split in ['train', 'val']: # pro trénovací a validační data
        losses = torch.zeros(eval_iters) # inicializuje tensor pro ukládání ztrát
        for k in range(eval_iters): # pro každou evaluační iteraci
            X, Y = get_batch(split) # získá dávku dat
            logits, loss = model(X, Y) # vypočítá logity a ztrátu
            losses[k] = loss.item() # uloží hodnotu ztráty
        out[split] = losses.mean() # průměrná ztráta pro daný split
    model.train() # přepne model zpět do trénovacího módu
    return out

class Head(nn.Module): # definice jedné hlavy self-attention
    """ jedna hlava self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # lineární transformace pro klíče
        self.query = nn.Linear(n_embd, head_size, bias=False) # lineární transformace pro dotazy
        self.value = nn.Linear(n_embd, head_size, bias=False) # lineární transformace pro hodnoty
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # maska pro dolní trojúhelníkovou matici

        self.dropout = nn.Dropout(dropout) # dropout vrstva

    def forward(self, x):
        B,T,C = x.shape # rozměry vstupu
        k = self.key(x)   # aplikace transformace na klíče
        q = self.query(x) # aplikace transformace na dotazy
        # výpočet skóre pozornosti ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # výpočet skóre pozornosti
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # aplikace masky
        wei = F.softmax(wei, dim=-1) # aplikace softmax pro normalizaci
        wei = self.dropout(wei) # aplikace dropoutu
        # provede váženou agregaci hodnot
        v = self.value(x) # aplikace transformace na hodnoty
        out = wei @ v # vážená agregace
        return out

class MultiHeadAttention(nn.Module): # definice vícehlavé pozornosti
    """ více hlav self-attention paralelně """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # seznam hlav pozornosti
        self.proj = nn.Linear(n_embd, n_embd) # lineární projekce výstupů
        self.dropout = nn.Dropout(dropout) # dropout vrstva

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # spojí výstupy všech hlav
        out = self.dropout(self.proj(out)) # aplikace projekce a dropoutu
        return out

# 3 ---------------------------------------------------------------------------------

class FeedFoward(nn.Module):
    """ jednoduchá lineární vrstva následovaná nelinearitou """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # zvýší dimenzi vstupu čtyřnásobně
            nn.ReLU(), # aplikuje ReLU aktivaci
            nn.Linear(4 * n_embd, n_embd), # sníží dimenzi zpět na původní velikost
            nn.Dropout(dropout), # aplikuje dropout pro redukci přeučení
        )

    def forward(self, x):
        return self.net(x) # projde vstup přes definovanou síť

class Block(nn.Module):
    """ Blok transformeru: komunikace následovaná výpočtem """

    def __init__(self, n_embd, n_head):
        # n_embd: dimenze vložení, n_head: počet hlav, které chceme
        super().__init__()
        head_size = n_embd // n_head # vypočítá velikost jedné hlavy
        self.sa = MultiHeadAttention(n_head, head_size) # vícehlavá pozornost
        self.ffwd = FeedFoward(n_embd) # feedforward síť
        self.ln1 = nn.LayerNorm(n_embd) # normalizace první vrstvy
        self.ln2 = nn.LayerNorm(n_embd) # normalizace druhé vrstvy

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # aplikuje self-attention a residuální spojení
        x = x + self.ffwd(self.ln2(x)) # aplikuje feedforward síť a residuální spojení
        return x

# 4 ---------------------------------------------------------------------------------

# velmi jednoduchý bigramový model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # každý token přímo čte logity pro další token z tabulky vyhledávání
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vložení pro tokeny
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # vložení pro pozice
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # sekvence transformerových bloků
        self.ln_f = nn.LayerNorm(n_embd) # normalizace poslední vrstvy
        self.lm_head = nn.Linear(n_embd, vocab_size) # lineární vrstva na konec

    def forward(self, idx, targets=None):
        B, T = idx.shape # rozměry vstupu

        # idx a targets jsou oba tenzory (B,T) celých čísel
        tok_emb = self.token_embedding_table(idx) # (B,T,C) vložení tokenů
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) vložení pozic
        x = tok_emb + pos_emb # (B,T,C) kombinace vložení tokenů a pozic
        x = self.blocks(x) # (B,T,C) prochází transformerovými bloky
        x = self.ln_f(x) # (B,T,C) normalizace
        logits = self.lm_head(x) # (B,T,vocab_size) vypočítá logity pro každý token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # přeformátuje logity pro výpočet ztráty
            targets = targets.view(B*T) # přeformátuje cíle pro výpočet ztráty
            loss = F.cross_entropy(logits, targets) # vypočítá ztrátu cross-entropy

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx je pole (B, T) indexů v aktuálním kontextu
        for _ in range(max_new_tokens): # pro každý nový token
            # ořízne idx na poslední block_size tokenů
            idx_cond = idx[:, -block_size:]
            # získá predikce
            logits, loss = self(idx_cond)
            # zaměří se pouze na poslední časový krok
            logits = logits[:, -1, :] # stává se (B, C)
            # aplikuje softmax pro získání pravděpodobností
            probs = F.softmax(logits, dim=-1) # (B, C)
            # vzorkuje z distribuce
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # připojí vzorkovaný index k běžící sekvenci
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# 5 ---------------------------------------------------------------------------------

model = BigramLanguageModel() # inicializuje model bigramového jazykového modelu
m = model.to(device) # přesune model na zařízení (GPU nebo CPU)
# vypíše počet parametrů v modelu
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # spočítá a vypíše celkový počet parametrů v milionech

# vytvoří PyTorch optimalizátor
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # používá AdamW optimalizátor s definovanou rychlostí učení

for iter in range(max_iters): # hlavní trénovací smyčka

    # občas vyhodnotí ztrátu na trénovacích a validačních datech
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss() # odhadne ztrátu
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # vypíše ztráty

    # vybere dávku dat
    xb, yb = get_batch('train') # získá dávku trénovacích dat

    # vyhodnotí ztrátu
    logits, loss = model(xb, yb) # spočítá logity a ztrátu pro dávku
    optimizer.zero_grad(set_to_none=True) # vynuluje gradienty optimalizátoru
    loss.backward() # provede backpropagation
    optimizer.step() # aktualizuje váhy modelu


# generování z modelu !!!!!!!!
context = torch.zeros((1, 1), dtype=torch.long, device=device) # inicializuje kontext pro generování
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist())) # dekóduje a vypíše vygenerovaný text

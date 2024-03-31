# Zde je podrobný popis programu s vysvětlením každé části v češtině:


# Import knihoven PyTorch pro práci s neuronovými sítěmi a tensorovými operacemi
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparametry modelu
batch_size = 64 # Kolik nezávislých sekvencí bude zpracováváno paralelně
block_size = 256 # Maximální délka kontextu pro predikce
max_iters = 5000 # Maximální počet iterací trénování
eval_interval = 500 # Jak často hodnotit ztrátu
learning_rate = 3e-4 # Rychlost učení
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Zařízení pro výpočty (GPU/CPU)
eval_iters = 200 # Počet iterací pro odhad ztráty
n_embd = 384 # Velikost vektorové reprezentace (embedding)
n_head = 6 # Počet hlav v multi-head attention
n_layer = 6 # Počet vrstev transformátoru
dropout = 0.2 # Pravděpodobnost vynechání neuronu při trénování (pro regulaci)

# Nastavení náhodného seedu pro reprodukovatelnost výsledků
torch.manual_seed(1337)

# Načtení textových dat
with open('rozmarne leto.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Vytvoření slovníku unikátních znaků v textu
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Mapování znaků na celá čísla a zpět
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # Encoder: převede řetězec na seznam celých čísel
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: převede seznam celých čísel na řetězec

# Rozdělení dat na trénovací a validační množinu
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # Prvních 90% dat pro trénink, zbytek pro validaci
train_data = data[:n]
val_data = data[n:]

# Funkce pro načtení dávky dat
def get_batch(split):
    # Vygeneruje malou dávku dat vstupů x a cílů y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Funkce pro odhad ztráty bez provádění zpětného šíření
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Přepne model do evaluačního módu
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Přepne model zpět do trénovacího módu
    return out

# Definice tříd pro komponenty modelu (Head, MultiHeadAttention, FeedForward, Block, BigramLanguageModel)
# Tyto třídy implementují architekturu transformátoru a mechanismus sebe-pozornosti (self-attention)

# Třída pro jazykový model BigramLanguageModel
class BigramLanguageModel(nn.Module):
    # Inicializace modelu, definice vrstev a parametrů
    def __init__(self):
        super().__init__()
        # Definice vrstev modelu
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Normální vrstva na konci
        self.lm_head = nn.Linear(n_embd, vocab_size) # Výstupní vrstva

    # Funkce pro dopředný průchod (forward pass)
    def forward(self, idx, targets=None):
        # idx a targets jsou tenzory celých čísel velikosti (B,T)
        tok_emb = self.token_embedding_table(idx) # Vektorové reprezentace tokenů
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Vektorové reprezentace pozic
        x = tok_emb + pos_emb # Součet token a pozicových embeddingů
        x = self.blocks(x) # Průchod skrze bloky transformátoru
        x = self.ln_f(x) # Aplikace normální vrstvy
        logits = self.lm_head(x) # Výpočet logitů pro každý token

        if targets is None:
            loss = None
        else:
            # Výpočet ztráty, pokud jsou zadány cíle
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Funkce pro generování textu
    def generate(self, idx, max_new_tokens):
        # idx je pole indexů aktuálního kontextu
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Oříznutí idx na poslední block_size tokenů
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # Zaměření pouze na poslední časový krok
            probs = F.softmax(logits, dim=-1) # Aplikace softmax
            idx_next = torch.multinomial(probs, num_samples=1) # Vzorkování z distribuce
            idx = torch.cat((idx, idx_next), dim=1) # Přidání vzorkovaného indexu do sekvence
        return idx

# Inicializace modelu, přesun na zařízení, výpis počtu parametrů
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Vytvoření optimalizátoru
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Hlavní trénovací smyčka
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train') # Načtení dávky dat
    logits, loss = model(xb, yb) # Výpočet ztráty
    optimizer.zero_grad(set_to_none=True) # Reset gradientů
    loss.backward() # Zpětné šíření
    optimizer.step() # Aktualizace parametrů modelu

# Generování textu z modelu
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


# Tento program implementuje jazykový model pomocí architektury transformátoru. Model se učí na základě textových dat a je schopen generovat text tím, že predikuje následující tokeny v sekvenci na základě předchozích tokenů. Hyperparametry,
```python
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
```
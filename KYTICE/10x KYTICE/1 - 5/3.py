```python
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
    """ Blok transformátoru: komunikace následovaná výpočtem """

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
```
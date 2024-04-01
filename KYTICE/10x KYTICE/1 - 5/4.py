```python
# velmi jednoduchý bigramový model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # každý token přímo čte logity pro další token z tabulky vyhledávání
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vložení pro tokeny
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # vložení pro pozice
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # sekvence transformátorových bloků
        self.ln_f = nn.LayerNorm(n_embd) # normalizace poslední vrstvy
        self.lm_head = nn.Linear(n_embd, vocab_size) # lineární vrstva na konec

    def forward(self, idx, targets=None):
        B, T = idx.shape # rozměry vstupu

        # idx a targets jsou oba tenzory (B,T) celých čísel
        tok_emb = self.token_embedding_table(idx) # (B,T,C) vložení tokenů
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) vložení pozic
        x = tok_emb + pos_emb # (B,T,C) kombinace vložení tokenů a pozic
        x = self.blocks(x) # (B,T,C) prochází transformátorovými bloky
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
```
```python
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

# generování z modelu
context = torch.zeros((1, 1), dtype=torch.long, device=device) # inicializuje kontext pro generování
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist())) # dekóduje a vypíše vygenerovaný text
```
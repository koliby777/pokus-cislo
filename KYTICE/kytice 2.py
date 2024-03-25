import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1  # Increased dropout for regularization

torch.manual_seed(1337)

# Load data
with open('kytice.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model definition and other components (omitted for brevity)


# Training loop setup
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=0)  # Cosine Annealing Scheduler
scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

# Training Loop
model.train()
for it in range(max_iters):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)  # Efficient zeroing of gradients

        with torch.cuda.amp.autocast():  # Mixed precision context
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        scaler.scale(loss).backward()  # Scale the loss to adjust for mixed precision
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)  # Optimizer step with scaled gradients
        scaler.update()  # Update the scale for next iteration

        scheduler.step()  # Update learning rate

        if batch_idx % eval_interval == 0:
            print(f"Iteration {it}, Loss: {loss.item()}")

            # Evaluation logic here (omitted for brevity)

# Save the model after training
torch.save(model.state_dict(), 'model.pth')

print("Training complete")

def generate_text(model, start_string, generation_length=100):
    """
    Generates text from a trained model.

    Parameters:
    - model: The trained PyTorch model.
    - start_string: Seed text to start the generation.
    - generation_length: Number of characters to generate.

    Returns:
    - The generated text.
    """
    model.eval()  # Set the model to eval mode
    generated_text = start_string
    input_ids = torch.tensor([encode(start_string)], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(generation_length):
            outputs = model(input_ids)
            predictions = outputs[:, -1, :]  # Take the last time step's output
            predicted_id = torch.argmax(predictions, axis=-1)
            generated_char = decode(predicted_id.cpu().numpy())[0]

            # Append the predicted character to the generated text
            generated_text += generated_char

            # Update the input_ids to contain the newly generated character
            input_ids = torch.cat((input_ids, predicted_id.unsqueeze(0)), dim=1)

    return generated_text

# Example usage:
start_string = "To be or not to be"
generation_length = 200  # Number of characters to generate
generated_text = generate_text(model, start_string, generation_length)
print("Generated Text:\n", generated_text)


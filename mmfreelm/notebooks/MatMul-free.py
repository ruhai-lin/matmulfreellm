import torch
import torch.nn as nn
from torch.nn import functional as F
import math
print(torch.cuda.is_available())
# Load the dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the unique characters that occur in this text
chars = sorted(list(set(text)))
dict_size = len(chars)
embedding_dim = 1024
batch_size = 1
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
eval_interval = 300
max_iters = 3000

# ---------------------------------------------------------------------------------------------------------

# Create a mapping from char to int
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------------------------------------------------------------------------------------------

class EmbeddingLayer(nn.Module):
    def __init__(self, dict_size=dict_size, embedding_dim=embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(dict_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# ---------------------------------------------------------------------------------------------------------

class MLGRU_Linear(nn.Module):
    def __init__(self, embedding_dim=embedding_dim, eps=1e-6):
        super(MLGRU_Linear, self).__init__()
        self.eps = eps
        self.embedding_dim = embedding_dim

        self.linear_f = nn.Linear(embedding_dim, embedding_dim)
        self.linear_c = nn.Linear(embedding_dim, embedding_dim)
        self.linear_g = nn.Linear(embedding_dim, embedding_dim)

        self.weight_g_in = nn.Parameter(torch.rand(self.embedding_dim))
        self.weight_g_f = nn.Parameter(torch.rand(self.embedding_dim))
        self.weight_g_c = nn.Parameter(torch.rand(self.embedding_dim))
        self.weight_g_g = nn.Parameter(torch.rand(self.embedding_dim))
        self.weight_g_norm = nn.Parameter(torch.rand(self.embedding_dim))

        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()

    def rms_norm(self, x, weight_g):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (x * weight_g)

    def forward(self, x):
        x = self.rms_norm(x, self.weight_g_in)
        x_f_in = self.rms_norm(x, self.weight_g_f)
        x_c_in = self.rms_norm(x, self.weight_g_c)
        x_g_in = self.rms_norm(x, self.weight_g_g)

        x_f = self.sigmoid(self.linear_f(x_f_in))
        x_c = self.silu(self.linear_c(x_c_in))
        x_g = self.rms_norm(self.linear_g(x_g_in), self.weight_g_norm)

        return x_f, x_c, x_g

# ---------------------------------------------------------------------------------------------------------

class MLGRU_Hidden(nn.Module):
    def __init__(self, embedding_dim=embedding_dim):
        super(MLGRU_Hidden, self).__init__()
        self.embedding_dim = embedding_dim
        self.weight_l = nn.Parameter(torch.rand(self.embedding_dim))

        self.softmax = nn.Softmax(dim=0)

    def lower_bound_fn(self):
        lb = self.softmax(self.weight_l).cumsum(dim=0) - self.weight_l[0]
        lb = lb.unsqueeze(0)  # Adjust shape to (1, embedding_dim)
        return lb

    def forward(self, x_f, x_c, h_t):
        lb = self.lower_bound_fn()
        x_f = lb + (1 - lb) * x_f
        h_t = x_f * h_t + (1 - x_f) * x_c
        return h_t

# ---------------------------------------------------------------------------------------------------------

class MLGRU_Output(nn.Module):
    def __init__(self, embedding_dim=embedding_dim, eps=1e-6):
        super(MLGRU_Output, self).__init__()
        self.embedding_dim = embedding_dim
        self.weight_g_o = nn.Parameter(torch.rand(self.embedding_dim))
        self.linear_o = nn.Linear(embedding_dim, embedding_dim)
        self.eps = eps

    def rms_norm(self, x, weight_g):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight_g

    def forward(self, h_t, x_g):
        x_o_tmp = x_g * h_t * h_t.sigmoid()
        x_o = self.linear_o(self.rms_norm(x_o_tmp, self.weight_g_o))
        return x_o

# ---------------------------------------------------------------------------------------------------------

class GLU(nn.Module):
    def __init__(self, embedding_dim=embedding_dim, hidden_ratio=4, eps=1e-6):
        super(GLU, self).__init__()
        self.eps = eps
        self.embedding_dim = embedding_dim
        self.hidden_ratio = hidden_ratio

        self.intermediate_size = int(self.embedding_dim * self.hidden_ratio * 2 / 3)
        self.intermediate_size = 256 * ((self.intermediate_size + 256 - 1) // 256)
        self.double_dim = self.intermediate_size * 2

        self.weight_g = nn.Parameter(torch.rand(self.embedding_dim))
        self.weight_g_o = nn.Parameter(torch.rand(self.embedding_dim))
        self.weight_d = nn.Parameter(torch.rand(self.intermediate_size))

        self.linear_u = nn.Linear(self.embedding_dim, self.double_dim)
        self.linear_n = nn.Linear(self.intermediate_size, self.embedding_dim)

        self.silu = nn.SiLU()

    def rms_norm(self, x, weight_g):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight_g

    def forward(self, x_o):
        x_o = self.linear_u(self.rms_norm(self.rms_norm(x_o, self.weight_g), self.weight_g_o))

        x_o_1 = self.silu(x_o[:, :self.intermediate_size])
        x_o_2 = x_o[:, self.intermediate_size:]

        x_d = self.linear_n(self.rms_norm((x_o_1 * x_o_2), self.weight_d))
        return x_d

# ---------------------------------------------------------------------------------------------------------

class OutputLayer(nn.Module):
    def __init__(self, dict_size=dict_size, embedding_dim=embedding_dim):
        super(OutputLayer, self).__init__()
        self.output = nn.Linear(embedding_dim, dict_size)

    def forward(self, x):
        logits = self.output(x)
        return logits

# ---------------------------------------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, embedding_dim, num_blocks, dict_size):
        super(Model, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.dict_size = dict_size

        self.embedding_layer = EmbeddingLayer(dict_size=self.dict_size, embedding_dim=embedding_dim)

        # Using nn.ModuleList to store sub-layers
        self.mlgru_linear_layers = nn.ModuleList()
        self.mlgru_hidden_layers = nn.ModuleList()
        self.mlgru_output_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()

        self.output_layer = OutputLayer(dict_size=self.dict_size, embedding_dim=embedding_dim)

        for _ in range(self.num_blocks):
            self.mlgru_linear_layers.append(
                MLGRU_Linear(embedding_dim=embedding_dim, eps=1e-6)
            )
            self.mlgru_hidden_layers.append(
                MLGRU_Hidden(embedding_dim=embedding_dim)
            )
            self.mlgru_output_layers.append(
                MLGRU_Output(embedding_dim=embedding_dim)
            )
            self.glu_layers.append(
                GLU(embedding_dim=embedding_dim)
            )

    def forward(self, x, targets=None):
        B, T = x.size()
        outputs = []

        # Initialize hidden states at the beginning of each forward pass
        h_t_list = [torch.zeros(B, self.embedding_dim, device=x.device) for _ in range(self.num_blocks)]

        for step in range(T):
            # Get embeddings for the current time step
            y_t = self.embedding_layer(x[:, step])

            for i in range(self.num_blocks):
                # MLGRU Linear layer
                x_f, x_c, x_g = self.mlgru_linear_layers[i](y_t)

                # MLGRU Hidden layer
                h_t = self.mlgru_hidden_layers[i](x_f, x_c, h_t_list[i])

                # Update hidden state
                h_t_list[i] = h_t

                # MLGRU Output layer
                x_o = self.mlgru_output_layers[i](h_t, x_g)

                # First residual connection
                x_o = y_t + x_o

                # GLU layer
                x_d = self.glu_layers[i](x_o)

                # Second residual connection, update y_t
                y_t = x_o + x_d

            # Output layer, get logits
            logits = self.output_layer(y_t)
            outputs.append(logits.unsqueeze(1))  # logits shape: (B, 1, dict_size)

        # Concatenate outputs over time
        logits = torch.cat(outputs, dim=1)  # logits shape: (B, T, dict_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# ---------------------------------------------------------------------------------------------------------

# Initialize the model
model = Model(
    embedding_dim=embedding_dim,
    num_blocks=2,  # Adjusted for testing; increase as needed
    dict_size=dict_size
).to(device)  # Move the model to the appropriate device (GPU if available)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initialize context on the correct device
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_indices))

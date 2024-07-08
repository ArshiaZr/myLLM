import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import notebook
import pickle
import os


# Hyperparameters
batch_size = 64
block_size = 128
n_embd = 384
n_decoders = 8
n_head = 8
dropout = 0.2


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        #perform the weighted aggregation of the values
        v = self.values(x)
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        
        # self-attention layer
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
        
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, device="cpu"):
        super().__init__()
        self.device = device
        # Token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position embeddings
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Decoder layers
        self.decoders = nn.Sequential(*[Decoder(n_embd, n_head=n_head) for _ in range(n_decoders)])
        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        # Final linear layer at the end
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.train_data = None
        self.val_data = None
        
        # Initialize weights properly to converge better
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            
    def forward(self, index, targets=None):
        B, T = index.shape
        
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.decoders(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

    @torch.no_grad()
    def estimate_loss(self, eval_iters=100):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data  # Selects the dataset based on the 'split' argument ('train' or 'val').
        
        # Randomly selects starting indices for the batch of sequences.
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # Extracts sequences of length 'block_size' starting from the random indices for the input data.
        x = torch.stack([data[i:i+block_size] for i in ix])
        
        # Extracts sequences of length 'block_size' starting from the next position after the input sequences for the target data.
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        # Moves the input and target tensors to the specified device (CPU or GPU).
        x, y = x.to(self.device), y.to(self.device)
        return x, y  # Returns the input and target tensors.
    
    def set_train_data(self, train_data):
        self.train_data = train_data

    def set_val_data(self, val_data):
        self.val_data = val_data

    def _train(self, epochs, learning_rate, eval_iters=100):
        assert self.train_data is not None, "train_data must be provided"
        assert self.val_data is not None, "val_data must be provided"

        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)
        for epoch in notebook.tqdm(range(epochs)):
            if epoch % eval_iters == 0:
                losses = self.estimate_loss(eval_iters=eval_iters)
                print(f"Epoch: {epoch}, train_loss: {losses['train']:.3f}, val_loss: {losses['val']:.3f}")
            
            xb, yb = self.get_batch('train')
        
            logits, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return loss.item()
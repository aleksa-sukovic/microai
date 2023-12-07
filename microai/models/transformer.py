import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    vocab_size: int = field(
        default=4e3,
        metadata={"help": "Size of the vocabulary"})
    d_model: int = field(
        default=768,
        metadata={"help": "Dimension of the model, including attention and feed-forward layers"})
    num_heads: int = field(
        default=12,
        metadata={"help": "Number of attention heads"})
    context_size: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to consider as context"})
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"})
    decoder_layers: int = field(
        default=12,
        metadata={"help": "Number of decoder layers"})


class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model

    def forward(self, x):
        _, T = x.shape
        embd = torch.zeros((T, self.d_model))
        div = torch.pow(1e4, torch.arange(0, self.d_model, 2) / self.d_model)
        pos = torch.arange(0, T).unsqueeze(1)
        embd[:, 0::2] = torch.sin(pos / div)
        embd[:, 1::2] = torch.cos(pos / div)
        return embd.to(x.device)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.l1 = nn.Linear(config.d_model, config.d_model)
        self.l2 = nn.Linear(config.d_model, config.d_model)
        self.drop = nn.Dropout(config.dropout) if config.dropout else None

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.drop(x) if self.drop else x
        return x


class Attention(nn.Module):
    def __init__(self, embd_dim, config: TransformerConfig, masked: bool = True):
        super().__init__()
        self.masked = masked
        self.d_model = config.d_model

        self.K = nn.Linear(config.d_model, embd_dim, bias=False)
        self.V = nn.Linear(config.d_model, embd_dim, bias=False)
        self.Q = nn.Linear(config.d_model, embd_dim, bias=False)

        mask = torch.ones((config.context_size, config.context_size))
        mask = torch.tril(mask) if masked else mask
        self.register_buffer("mask", mask)

    def forward(self, x):
        _, T, _ = x.shape

        Q = self.Q(x) # B x T x d_model
        K = self.K(x) # B x T x d_model
        V = self.V(x) # B x T x d_model

        w = Q @ K.transpose(1, 2)                                # B x T x T
        w = w / torch.sqrt(torch.tensor(self.d_model))           # scaled dot product
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # prevent attending to future tokens, if masked
        w = F.softmax(w, dim=-1)                                 # normalize scores
        w = w @ V                                                # obtain values

        return w                                                 # B x T x d_model


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig, masked: bool = True):
        super().__init__()
        self.heads = nn.ModuleList([Attention(config.d_model // config.num_heads, config, masked) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        w = [head(x) for head in self.heads]
        w = torch.cat(w, dim=-1)
        return self.proj(w)


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config, masked=True)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.ln1(self.attention(x))
        x = x + self.ln2(self.ff(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embd_input = nn.Embedding(config.vocab_size, config.d_model)
        self.embd_pos = PositionalEmbedding(config)
        self.decoders = nn.Sequential(*[DecoderBlock(config) for _ in range(config.num_heads)])
        self.linear = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout else None

    def forward(self, x):
        x = self.embd_input(x) + self.embd_pos(x)
        x = self.decoders(x)
        x = self.dropout(x) if self.dropout else x
        x = self.linear(x)
        return x

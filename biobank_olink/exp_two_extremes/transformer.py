import math
from functools import partial
from typing import Optional

import torchtuples as tt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from biobank_olink.constants import SEED


class FeedForwardBlock(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # w1 and b1
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, n_embd) --> (batch, seq_len, d_ff) --> (batch, seq_len, n_embd)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class InputEmbeddings(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, n_embd)
        # Multiply by sqrt(n_embd) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.n_embd)


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, n_embd)
        pe = torch.zeros(seq_len, n_embd)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (n_embd)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd)
        )  # (n_embd / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / n_embd))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / n_embd))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, n_embd)
        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe.requires_grad_(False)  # (batch, seq_len, n_embd)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, n_embd: int, h: int, dropout: float) -> None:
        super().__init__()
        self.n_embd = n_embd  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure n_embd is divisible by h
        assert n_embd % h == 0, "n_embd is not divisible by h"
        self.d_k = n_embd // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(n_embd, n_embd, bias=False)  # Wq
        self.w_k = nn.Linear(n_embd, n_embd, bias=False)  # Wk
        self.w_v = nn.Linear(n_embd, n_embd, bias=False)  # Wv
        self.w_o = nn.Linear(n_embd, n_embd, bias=False)  # Wo
        self.dropout = dropout

    def forward(self, q, k, v):
        query = self.w_q(q)  # (batch, seq_len, n_embd) --> (batch, seq_len, n_embd)
        key = self.w_k(k)  # (batch, seq_len, n_embd) --> (batch, seq_len, n_embd)
        value = self.w_v(v)  # (batch, seq_len, n_embd) --> (batch, seq_len, n_embd)
        # (batch, seq_len, n_embd) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # Calculate attention
        x = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, n_embd)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # Multiply by Wo
        # (batch, seq_len, n_embd) --> (batch, seq_len, n_embd)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            features: int,
            self_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            src_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            projection_layer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def forward(self, x: torch.Tensor):
        # (batch, in_feats, n_embd)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self._encode(x)
            # (batch, in_feats, vocab_size) -> (batch, in_feats * vocab_size)
            x = x.view(x.size(0), -1)
        x = self.projection_layer(x)
        return torch.nn.functional.sigmoid(x)

    def _encode(self, src):
        # (batch, seq_len, n_embd)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src)

    @classmethod
    def build(
            cls,
            in_feats: int,
            vocab_size: int,
            n_layer: int,
            n_head: int,
            n_embd: int,
            dropout: float = 0.0,
    ):
        embd = InputEmbeddings(n_embd, vocab_size)
        src_pos = PositionalEncoding(n_embd, in_feats, dropout)
        # Create the encoder
        encoder = Encoder(
            n_embd,
            nn.ModuleList(
                [
                    EncoderBlock(
                        n_embd,
                        MultiHeadAttentionBlock(n_embd, n_head, dropout),
                        FeedForwardBlock(n_embd, dropout),
                        dropout,
                    )
                    for _ in range(n_layer)
                ]
            ),
        )
        # Create the projection layer
        projection_layer = nn.Linear(in_feats * n_embd, 1)
        # Create the transformer
        transformer = Transformer(encoder, embd, src_pos, projection_layer)
        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer


def get_transformer(
        in_feats: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        **kwargs
) -> tt.Model:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    net = Transformer.build(in_feats, vocab_size, n_layer, n_head, n_embd, dropout)
    # net = torch.compile(net)
    optimizer = tt.optim.AdamW(lr=learning_rate)
    model = tt.Model(net, loss=nn.BCELoss(), optimizer=optimizer, device=device)

    model.predict_proba = partial(model.predict_net, batch_size=1024, to_cpu=True)
    return model


class Tokenizer:
    bins: Optional[dict] = None
    n_bins: int

    def __init__(self, n_bins: int):
        self.n_bins = n_bins

    @property
    def num_tokens(self):
        return self.n_bins + 1

    def fit_transform(self, x: pd.DataFrame):
        self.bins = {}
        return x.transform(self._quantile_fit_transform).fillna(self.n_bins).astype(np.int64)

    def _quantile_fit_transform(self, values: pd.Series):
        binned, bins = pd.qcut(values, q=self.n_bins, labels=False, retbins=True, duplicates="drop")
        self.bins[values.name] = bins
        return binned

    def transform(self, x: pd.DataFrame):
        if self.bins is None:
            raise RuntimeError("Call fit_transform() first")
        return x.transform(self._quantile_transform).fillna(self.n_bins).astype(np.int64)

    def _quantile_transform(self, values: pd.Series):
        bins = self.bins[values.name]
        return pd.cut(values.clip(bins[0], bins[-1]), bins=bins, labels=False, include_lowest=True)

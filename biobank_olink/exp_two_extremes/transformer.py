from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt

from biobank_olink.constants import SEED


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_head: int,
            d_ff: int,
            n_layer: int,
            vocab_size: int,
            bias: bool = False,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_enc = PositionalEncoding(n_embd, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            bias=bias,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.output_layer = nn.Linear(n_embd, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # input: (batch, seq_len)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.embedding(x)  # -> (batch, seq_len, n_embd)
            x = self.pos_enc(x)  # -> (batch, seq_len, n_embd)
            x = self.transformer_encoder(x)  # -> (batch, seq_len, vocab_size)
            x = x.mean(dim=1)  # -> (batch, vocab_size)
        x = self.output_layer(x)  # -> (batch, 1)
        return self.sigmoid(x)


def get_transformer(
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
        learning_rate: float = 1e-3,
        device: str = "cuda",
        **kwargs
) -> tt.Model:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    net = Transformer(n_embd, n_head, d_ff, n_layer, vocab_size, bias, dropout)
    # net = torch.compile(net)
    optimizer = tt.optim.AdamW(lr=learning_rate)
    model = tt.Model(net, loss=nn.BCELoss(), optimizer=optimizer, device=device)
    model.predict_proba = partial(model.predict_net, batch_size=128, to_cpu=True)
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

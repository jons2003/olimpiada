"""
Transformer-based V(s) head — self-attention over tokens, mean/max pool, MLP head.
Inputs:
  dense:         (B, N, DENSE_DIM)  float32
  content_value: (B, N)             int64
  target_value:  (B, N)             int64
Output:
  v: (B,) float32 — predicted distance to solved (in number of actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DENSE_DIM, VALUE_VOCAB

HIDDEN = 256
VALUE_EMB_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1


class ValueNet(nn.Module):
    def __init__(
        self,
        hidden: int = HIDDEN,
        value_emb_dim: int = VALUE_EMB_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.content_value_emb = nn.Embedding(VALUE_VOCAB, value_emb_dim)
        self.target_value_emb = nn.Embedding(VALUE_VOCAB, value_emb_dim)

        in_dim = DENSE_DIM + 2 * value_emb_dim
        self.input_proj = nn.Linear(in_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        dense: torch.Tensor,
        content_value: torch.Tensor,
        target_value: torch.Tensor,
    ) -> torch.Tensor:
        cv = self.content_value_emb(content_value)
        tv = self.target_value_emb(target_value)
        x = torch.cat([dense, cv, tv], dim=-1)
        x = self.input_proj(x)                     # (B, N, hidden)
        x = self.transformer(x)                    # (B, N, hidden)
        mean_pool = x.mean(dim=1)                  # (B, hidden)
        max_pool, _ = x.max(dim=1)                 # (B, hidden)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*hidden)
        v = self.head(pooled).squeeze(-1)          # (B,)
        return F.softplus(v)

    @staticmethod
    def count_params(model) -> int:
        return sum(p.numel() for p in model.parameters())
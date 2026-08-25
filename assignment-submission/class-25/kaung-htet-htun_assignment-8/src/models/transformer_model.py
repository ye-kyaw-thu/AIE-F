#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/transformer_model.py - Transformer Encoder for MSL Recognition

Architecture (ViT-inspired for sequences)
─────────────────────────────────────────
Input (B, T, D=225)
  → Linear projection → d_model
  → [CLS] token prepended
  → Sinusoidal positional encoding
  → TransformerEncoder (N layers, multi-head self-attention + FFN)
  → CLS token output → LayerNorm → Dropout → FC → num_classes

References
──────────
  Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
  Dosovitskiy et al., "An Image is Worth 16×16 Words", ICLR 2021.
  Jiang et al., "Sign Language Recognition via Skeleton-Aware Multi-Model
    Ensemble", ICCV 2021.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding as in Vaswani et al. 2017.
    Supports sequences up to max_len.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SignTransformer(nn.Module):
    """
    Transformer encoder for sign language sequence classification.
    """

    def __init__(
        self,
        input_dim:          int,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 4,
        dim_feedforward:    int   = 1024,
        dropout:            float = 0.1,
        num_classes:        int   = 100,
        max_seq_len:        int   = 210,
        use_cls_token:      bool  = True,
        activation:         str   = 'gelu',
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.use_cls_token = use_cls_token
        self.d_model       = d_model

        # ── Input branch ──────────────────────────────────────────────────────
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.proj_norm  = nn.LayerNorm(d_model)

        # Learnable [CLS] token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        extra = 1 if use_cls_token else 0
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout, max_len=max_seq_len + extra)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = activation,
            batch_first     = True,
            norm_first      = True,       # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_encoder_layers,
            norm       = nn.LayerNorm(d_model),
            enable_nested_tensor = False,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head_norm    = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.fc           = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x:    torch.Tensor,              # (B, T, D)
        mask: Optional[torch.Tensor] = None,  # (B, T) – True where padded
    ) -> torch.Tensor:

        B = x.shape[0]

        # Project input
        x = self.input_norm(x)
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.proj_norm(x)

        # Prepend CLS token
        if self.use_cls_token:
            cls   = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x     = torch.cat([cls, x], dim=1)         # (B, T+1, d_model)
            if mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoder (mask: True = ignore)
        x = self.encoder(x, src_key_padding_mask=mask)

        # Pool: use CLS token output if available, else mean pool
        if self.use_cls_token:
            pooled = x[:, 0, :]                        # (B, d_model)
        else:
            if mask is not None:
                valid_mask = ~mask                     # True where valid
                lengths    = valid_mask.sum(1, keepdim=True).clamp(min=1).float()
                pooled     = (x * valid_mask.unsqueeze(-1)).sum(1) / lengths
            else:
                pooled = x.mean(dim=1)

        pooled = self.head_norm(pooled)
        pooled = self.head_dropout(pooled)
        return self.fc(pooled)

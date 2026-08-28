#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/bilstm.py - Bidirectional LSTM with Multi-Head Attention for MSL Recognition

Architecture
────────────
Input (B, T, D=225)
  → Input projection + Layer-Norm
  → Bidirectional LSTM × num_layers   (with variational dropout)
  → Additive / Dot-product Attention  (collapses T dimension)
  → Dropout → FC → num_classes

References
──────────
  Bahdanau et al., "Neural Machine Translation by Jointly Learning to
  Align and Translate" (additive attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over a sequence.

    Computes a context vector as a weighted sum of hidden states,
    where weights are proportional to exp(v · tanh(W h_t)).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden:  torch.Tensor,   # (B, T, H)
        mask:    Optional[torch.Tensor] = None,  # (B, T) – True where padded
    ) -> torch.Tensor:
        # Score: (B, T, 1)
        energy  = torch.tanh(self.W(hidden))
        scores  = self.v(energy).squeeze(-1)   # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (weights * hidden).sum(dim=1)           # (B, H)
        return context


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM classifier for sign language recognition.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        num_layers:  int,
        num_classes: int,
        dropout:     float  = 0.4,
        bidirectional: bool = True,
        attention_type: str = 'additive',   # 'additive' | 'dot'
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim    = hidden_dim

        # Input normalization + projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM stack (inter-layer dropout handled by nn.LSTM)
        self.lstm = nn.LSTM(
            input_size    = hidden_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * self.num_directions

        # Attention
        self.attention_type = attention_type
        if attention_type == 'additive':
            self.attention = AdditiveAttention(lstm_out_dim)
        # dot-product attention: use F.scaled_dot_product_attention

        # Output head
        self.norm    = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_out_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 to encourage long-term memory
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x:       torch.Tensor,              # (B, T, D)
        lengths: Optional[torch.Tensor] = None,  # (B,) true lengths
        mask:    Optional[torch.Tensor] = None,   # (B, T) pad mask
    ) -> torch.Tensor:

        # Input projection
        x = self.input_norm(x)
        x = self.input_proj(x)             # (B, T, H)

        # Pack for efficiency if lengths provided
        if lengths is not None:
            packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            # pad_packed_sequence trims output to the longest sequence in the
            # batch, which may be shorter than max_seq_len.  The incoming mask
            # still has shape (B, max_seq_len), so we must trim it to match.
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            actual_T    = lstm_out.size(1)
            if mask is not None:
                mask = mask[:, :actual_T]
        else:
            lstm_out, _ = self.lstm(x)     # (B, T, H*dirs)

        # Attention pooling
        if self.attention_type == 'additive':
            context = self.attention(lstm_out, mask)
        else:
            # Simple dot-product: query = mean of lstm_out
            q       = lstm_out.mean(dim=1, keepdim=True)      # (B, 1, H)
            scores  = torch.bmm(q, lstm_out.transpose(1, 2))  # (B, 1, T)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, lstm_out).squeeze(1) # (B, H)

        context = self.norm(context)
        context = self.dropout(context)
        return self.fc(context)


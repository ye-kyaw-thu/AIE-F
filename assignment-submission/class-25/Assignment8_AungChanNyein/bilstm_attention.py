"""
bilstm_attention.py - BiLSTM with multi-head self-attention pooling

Extends the baseline BiLSTM by replacing simple additive attention with
multi-head self-attention over the LSTM output sequence, followed by
attention-weighted pooling. Should help on longer/noisier sequences
(MSL videos range 50-387 frames) by letting the model focus on the
most discriminative frames per sign.
"""
import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=256, num_layers=3,
                 num_classes=556, dropout=0.4, num_heads=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2

        self.self_attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # additive pooling attention: learns which timesteps matter most
        self.pool_attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

    def forward(self, x, lengths=None, mask=None):
        # x: (B, T, input_dim)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # True where padded

        attn_out, _ = self.self_attn(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask,
        )
        x = self.attn_norm(lstm_out + attn_out)  # residual

        # attention-weighted pooling over time
        scores = self.pool_attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)  # (B, 2*hidden)

        return self.classifier(pooled)
"""
cnn_gru.py - 1D CNN feature extractor + GRU temporal model

CNN1D layers extract local spatial-temporal patterns across the 225-dim
keypoint feature axis (per-frame), then a GRU models longer-range
temporal dependencies across frames. CNN front-end is faster to train
than pure RNN approaches and adds a different inductive bias (local
pattern detection) compared to BiLSTM/Transformer/ST-GCN.
"""
import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    def __init__(self, input_dim=225, cnn_channels=128, gru_hidden=256,
                 gru_layers=2, num_classes=556, dropout=0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        gru_out_dim = gru_hidden * 2

        self.pool_attn = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.Tanh(),
            nn.Linear(gru_out_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim // 2, num_classes),
        )

    def forward(self, x, lengths=None, mask=None):
        # x: (B, T, input_dim) -> conv expects (B, C, T)
        x = x.transpose(1, 2)            # (B, input_dim, T)
        x = self.conv_block(x)           # (B, cnn_channels, T)
        x = x.transpose(1, 2)            # (B, T, cnn_channels)

        gru_out, _ = self.gru(x)         # (B, T, 2*gru_hidden)

        scores = self.pool_attn(gru_out).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (gru_out * weights).sum(dim=1)

        return self.classifier(pooled)
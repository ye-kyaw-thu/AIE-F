"""This module defines the BiLSTM-based emotion classification model architecture."""

import torch
import torch.nn as nn


class EmotionalBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=64,
        output_dim=6,
        num_layers=1,
        dropout=0.2,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # dropout in LSTM only applies when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)

        # concatenate last forward and backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_cat = torch.cat((h_forward, h_backward), dim=1)

        logits = self.fc(self.dropout(h_cat))
        return logits

"""This module defines the BiLSTM-based emotion classification model architecture."""

import torch
import torch.nn as nn


class EmotionalBiLSTM(nn.Module):
    ## CHANGE HERE: tune default model hyperparameters
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=64,
        output_dim=6,
        num_layers=1,
        dropout=0.2,
        pad_idx=0,
        use_attention=False,
    ):
        super().__init__()

        self.use_attention = use_attention

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # dropout in LSTM only applies when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        # attention layer: maps LSTM outputs to a single context vector
        if self.use_attention:
            self.attention = nn.Linear(hidden_dim * 2, 1)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)


    # forward pass
    def forward(self, x):
        # embed input tokens
        embedded = self.embedding(x)

        # pass through LSTM
        lstm_out, (h_n, _) = self.lstm(embedded)

        # apply attention if enabled
        if self.use_attention:
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attention_weights).sum(dim=1)
        else:
            # concatenate last forward and backward hidden states
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            context = torch.cat((h_forward, h_backward), dim=1)

        # pass through fully connected layer
        logits = self.fc(self.dropout(context))

        return logits
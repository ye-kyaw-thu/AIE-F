"""This module defines the BiLSTM-based emotion classification model architecture."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# custom attention layer (subclass of nn.Module)
class Attention(nn.Module):
    """
    - nn.Linear maps each timestep to one score.
    - masked_fill() replaces padded positions with a large negative value before softmax
    so those positions receive near-zero weight after softmax.
    - softmax turns scores into weights over the sequence.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    # attention module's own forward pass: pools all timesteps into one vector per row
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        scores = self.attn(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights


class EmotionalBiLSTM(nn.Module):
    ## CHANGE HERE: tune default model hyperparameters
    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        hidden_dim=256,
        output_dim=6,
        num_layers=2,
        dropout=0.2,
        pad_idx=0,
        use_attention=True,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.use_attention = use_attention
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim

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

        if self.use_attention:
            self.attention = Attention(hidden_dim)

        # dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)


    # full forward pass (separate from Attention.forward): embedding → lstm → pooling → classifier
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)

        # per-row lengths let the lstm stop after real tokens instead of stepping through padding to max_len
        if lengths is None:
            lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1)

        # at least one timestep per row so packing never receives an empty sequence
        lengths = lengths.clamp(min=1)

        # pack_padded_sequence expects this tensor on cpu
        lengths_cpu = lengths.detach().cpu().long()

        # pack uses each row's true length so short sequences skip extra padding timesteps in the lstm
        packed = pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)

        # apply attention if only enabled
        ## pad_packed_sequence restores a fixed-width matrix for attention pooling
        ## Attention's mask hides padding
        if self.use_attention:
            lstm_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
            mask = x != self.pad_idx
            context, _weights = self.attention(lstm_out, mask=mask)
        else:
            ## no attention: concatenate last lstm hidden states (forward and backward directions)
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            context = torch.cat((h_forward, h_backward), dim=1)

        # pass through fully connected layer
        logits = self.fc(self.dropout_layer(context))

        return logits

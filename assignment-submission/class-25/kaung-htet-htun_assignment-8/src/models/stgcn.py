#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/stgcn.py - Spatial-Temporal Graph Convolutional Network for MSL Recognition

Architecture
────────────
Input (B, 3, T, 75)   ← (batch, coords, frames, nodes)
  → ST-GCN blocks (spatial graph conv + temporal conv)
  → Global average pooling over T and V
  → FC → num_classes

The skeleton graph has 75 nodes:
  0–32   Pose  (MediaPipe Holistic pose)
  33–53  Left  hand
  54–74  Right hand

Edges:
  Pose        → MediaPipe body skeleton
  Left hand   → MediaPipe hand topology
  Right hand  → MediaPipe hand topology
  Cross edges → left wrist (15) ↔ left hand root (33)
                right wrist (16) ↔ right hand root (54)

References
──────────
  Yan et al., "Spatial Temporal Graph Convolutional Networks for
  Skeleton-Based Action Recognition", AAAI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


# ─── Skeleton graph definition ────────────────────────────────────────────────

def _build_adjacency() -> np.ndarray:
    """
    Build the 75×75 adjacency matrix for the MSL holistic skeleton.
    Returns a symmetric binary matrix (self-loops included via I later).
    """
    N = 75
    A = np.zeros((N, N), dtype=np.float32)

    def add_edges(edges):
        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1

    # MediaPipe Pose connections (33 nodes, 0-indexed)
    pose_edges = [
        (0,1),(1,2),(2,3),(3,7),
        (0,4),(4,5),(5,6),(6,8),
        (9,10),
        (11,12),(11,13),(13,15),(15,17),(17,19),(19,15),
        (15,21),(12,14),(14,16),(16,18),(18,20),(20,16),(16,22),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
        (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
    ]
    add_edges(pose_edges)

    # MediaPipe Hand connections – apply to both hands with offset
    hand_topo = [
        (0,1),(1,2),(2,3),(3,4),        # thumb
        (0,5),(5,6),(6,7),(7,8),        # index
        (0,9),(9,10),(10,11),(11,12),   # middle
        (0,13),(13,14),(14,15),(15,16), # ring
        (0,17),(17,18),(18,19),(19,20), # pinky
        (5,9),(9,13),(13,17),           # palm
    ]
    # Left hand: offset 33
    add_edges([(i+33, j+33) for i, j in hand_topo])
    # Right hand: offset 54
    add_edges([(i+54, j+54) for i, j in hand_topo])

    # Cross-body: wrist → hand root
    A[15, 33] = 1; A[33, 15] = 1   # left  wrist ↔ left  hand root
    A[16, 54] = 1; A[54, 16] = 1   # right wrist ↔ right hand root

    return A


def normalize_adjacency(A: np.ndarray) -> torch.Tensor:
    """
    Symmetric normalisation: D^{-1/2} (A + I) D^{-1/2}
    as used in GCN (Kipf & Welling, 2017).
    """
    A = A + np.eye(A.shape[0], dtype=np.float32)  # add self-loops
    degree = A.sum(axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = np.diag(d_inv_sqrt)
    return torch.from_numpy(D @ A @ D)


# ─── Graph convolution ────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """Single-partition spatial graph convolution."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer('A', A)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor, A: torch.Tensor = None) -> torch.Tensor:
        # x: (B, C, T, V)
        # A: optional scaled adjacency (V, V); falls back to self.A
        A_use = A if A is not None else self.A
        x = self.conv(x)                              # (B, C_out, T, V)
        x = torch.einsum('bctv,vw->bctw', x, A_use)
        return x


class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
      Spatial GCN → BN → ReLU → Temporal Conv (k×1) → BN → ReLU → Dropout
    with optional residual connection.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        A:            torch.Tensor,
        temporal_kernel_size: int   = 9,
        stride:       int   = 1,
        dropout:      float = 0.3,
        residual:     bool  = True,
    ):
        super().__init__()

        pad = (temporal_kernel_size - 1) // 2

        self.gcn = GraphConv(in_channels, out_channels, A)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel_size, 1),
                      stride=(stride, 1),
                      padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if residual and in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        elif residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor = None) -> torch.Tensor:
        res = x
        x   = self.gcn(x, A)   # pass scaled A through; gcn falls back to self.A if None
        x   = self.tcn(x)
        if self.residual is not None:
            x = x + self.residual(res)
        return self.relu(x)


# ─── Full ST-GCN model ────────────────────────────────────────────────────────

class STGCN(nn.Module):
    """
    Spatial-Temporal GCN for Myanmar Sign Language recognition.

    Input:  (B, 3, T, 75)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        num_classes:  int   = 100,
        num_nodes:    int   = 75,
        dropout:      float = 0.3,
        temporal_kernel_size: int = 9,
        edge_importance_weighting: bool = True,
    ):
        super().__init__()

        # Build and normalise adjacency
        A_raw  = _build_adjacency()               # (75, 75)
        A_norm = normalize_adjacency(A_raw)       # (75, 75) – FloatTensor

        # Learnable edge importance weights (per block)
        self.edge_importance_weighting = edge_importance_weighting

        # Data batch-norm on input
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # Channel progression: 64 → 64 → 64 → 128 → 128 → 128 → 256 → 256 → 256
        channels = [
            (in_channels, 64),
            (64, 64),
            (64, 64),
            (64, 128),
            (128, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 256),
        ]

        self.st_gcn_blocks = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(channels):
            stride = 2 if i in [3, 6] else 1    # downsample at stage transitions
            self.st_gcn_blocks.append(
                STGCNBlock(
                    c_in, c_out, A_norm,
                    temporal_kernel_size = temporal_kernel_size,
                    stride               = stride,
                    dropout              = dropout,
                )
            )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(A_norm))
                for _ in self.st_gcn_blocks
            ])

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(256, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, 75)
        B, C, T, V = x.shape

        # Data BN: flatten (C, V) → apply BN → restore
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, V, C, T)
        x = x.view(B, V * C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()  # (B, C, T, V)

        # ST-GCN blocks — compute scaled adjacency out-of-place so autograd
        # never sees an in-place modification of a registered buffer.
        for i, block in enumerate(self.st_gcn_blocks):
            if self.edge_importance_weighting:
                # Multiply base adjacency by learnable importance weights.
                # clamp keeps the graph non-negative (no inhibitory edges).
                # This is a pure out-of-place op; block.gcn.A is never modified.
                A_scaled = (block.gcn.A * self.edge_importance[i]).clamp(min=0)
                x = block(x, A_scaled)
            else:
                x = block(x)

        # Global average pooling
        x = self.pool(x)                             # (B, 256, 1, 1)
        x = x.view(B, -1)                            # (B, 256)
        x = self.dropout(x)
        return self.fc(x)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/__init__.py - Model factory for MSL Recognition
"""

import torch
import torch.nn as nn
from .bilstm import BiLSTM
from .transformer_model import SignTransformer
from .stgcn import STGCN


def build_model(model_type: str, cfg: dict, num_classes: int) -> nn.Module:
    """
    Instantiate a model from config.

    Args:
        model_type:  'bilstm' | 'transformer' | 'stgcn'
        cfg:         full config dict (from config.yaml)
        num_classes: number of sign classes

    Returns:
        nn.Module
    """
    mcfg = cfg['model']

    if model_type == 'bilstm':
        bc = mcfg['bilstm']
        model = BiLSTM(
            input_dim      = bc['input_dim'],
            hidden_dim     = bc['hidden_dim'],
            num_layers     = bc['num_layers'],
            num_classes    = num_classes,
            dropout        = bc['dropout'],
            bidirectional  = bc['bidirectional'],
            attention_type = bc.get('attention_type', 'additive'),
        )

    elif model_type == 'transformer':
        tc = mcfg['transformer']
        model = SignTransformer(
            input_dim          = tc['input_dim'],
            d_model            = tc['d_model'],
            nhead              = tc['nhead'],
            num_encoder_layers = tc['num_encoder_layers'],
            dim_feedforward    = tc['dim_feedforward'],
            dropout            = tc['dropout'],
            num_classes        = num_classes,
            max_seq_len        = tc['max_seq_len'],
            use_cls_token      = tc.get('use_cls_token', True),
        )

    elif model_type == 'stgcn':
        sc = mcfg['stgcn']
        model = STGCN(
            in_channels  = sc['in_channels'],
            num_classes  = num_classes,
            num_nodes    = sc['num_nodes'],
            dropout      = sc['dropout'],
            edge_importance_weighting = sc.get('edge_importance_weighting', True),
        )

    else:
        raise ValueError(f"Unknown model type: '{model_type}'. "
                         f"Choose from: bilstm, transformer, stgcn")

    return model


__all__ = ['build_model', 'BiLSTM', 'SignTransformer', 'STGCN']

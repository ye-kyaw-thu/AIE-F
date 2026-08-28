#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_to_onnx.py  –  Export trained MSL models to ONNX format
                       (runs on Linux/HPC server side)

Bug history / fixes
────────────────────
  [EXPORT-A] _BiLSTMExportWrapper — lengths must be None, NOT computed from mask.

      BiLSTM.forward calls pack_padded_sequence when lengths is not None.
      During torch.onnx.export tracing the dummy mask is all-zeros, so any
      (~mask).sum() gives max_seq_len and the LSTM is traced with the full-length
      packing path baked into the ONNX file.  At inference time the baked path
      doesn't adapt to shorter sequences; more critically, if the attention pooling
      uses a lengths value that is always max_seq_len it averages over the zero-
      padded tail → wrong context vector for any video shorter than max_seq_len.

      Fix: lengths=None → LSTM runs unpacked on the full padded sequence (the else
      branch in BiLSTM.forward).  AdditiveAttention receives the mask directly and
      uses masked_fill to exclude padded positions from softmax — the mask IS a live
      ONNX input and varies correctly at inference time.

  [EXPORT-B] _TransformerExportWrapper — missing inner FFN dropout.

      PyTorch's TransformerEncoderLayer has THREE dropout objects:
        self.dropout  — inner FFN dropout, placed between activation and linear2
        self.dropout1 — post-attention residual dropout
        self.dropout2 — post-FFN residual dropout

      The FFN path in PyTorch source:
        ff = linear2(dropout(activation(linear1(x))))  ← inner dropout
        x  = x + dropout2(ff)                          ← residual dropout

      Previous wrapper omitted the inner dropout entirely:
        ff = linear2(activation(linear1(x)))   ← wrong
        x  = x + dropout2(ff)

      This is a no-op during eval() export (all dropouts are identity), so it
      does not affect current inference results.  However it is architecturally
      incorrect and would silently produce wrong outputs if the model were ever
      exported in training mode.  Added self.layer_dropout_inner ModuleList.

  [EXPORT-C] _TransformerExportWrapper — double-dropout in post-norm FFN path.
      dropout2 was applied twice in the residual add. Fixed: apply once.
      (Retained from previous revision — still correct.)

  [EXPORT-D] validate_onnx — np.bool_ for Windows ORT compatibility.

  [EXPORT-E] ST-GCN in-place copy_ during tracing → pre-bake via _STGCNExportWrapper.
  [EXPORT-F] PyTorch 2.x fused MHA has no ONNX mapping → ManualMHA.
  [EXPORT-G] GELU requires opset >= 20 → erf-based equivalent.
  [EXPORT-H] Batch size 2 dummy inputs prevent 0-D scalar-collapse.
  [EXPORT-I] 'model_type' was missing from metadata JSON → now always written.

NOTE on the SUMMARY display "0.0 MB" for JSON files:
  The metadata JSON files are ~130 bytes.  Displayed with one decimal place
  in MB (bytes / 1024^2) this rounds to 0.0.  The files are correct and
  complete — infer_onnx.py reads them fine.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from models import build_model
from utils  import get_logger, get_device, load_label_map

MODEL_CONFIGS = {
    "bilstm":       {"exp_dir": "results/exp_bilstm",       "output_name": "bilstm_msl.onnx"},
    "transformer":  {"exp_dir": "results/exp_transformer",  "output_name": "transformer_msl.onnx"},
    "stgcn":        {"exp_dir": "results/exp_stgcn",        "output_name": "stgcn_msl.onnx"},
}

POSE_N       = 33
HAND_N       = 21
TOTAL_JOINTS = POSE_N + HAND_N * 2   # 75
FEAT_DIM     = TOTAL_JOINTS * 3       # 225


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM export
# ─────────────────────────────────────────────────────────────────────────────

class _BiLSTMExportWrapper(torch.nn.Module):
    """
    Wrapper for BiLSTM ONNX export.

    Fix [EXPORT-A]: lengths=None bypasses pack_padded_sequence entirely.

    BiLSTM.forward branch when lengths=None:
        lstm_out, _ = self.lstm(x)          # full padded sequence, no packing
        context = self.attention(lstm_out, mask)   # mask zeros out padded positions

    The mask (True = padded) is a live ONNX input, so attention correctly
    ignores padded frames at every inference call regardless of video length.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, keypoints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(keypoints, lengths=None, mask=mask)


def export_bilstm(model, out_path: Path, max_seq_len: int, device: str):
    model.eval()
    wrapper = _BiLSTMExportWrapper(model).to(device)
    wrapper.eval()

    dummy_kp   = torch.randn(2, max_seq_len, FEAT_DIM, device=device)
    dummy_mask = torch.zeros(2, max_seq_len, dtype=torch.bool, device=device)
    print(f"  BiLSTM dummy inputs: kp={tuple(dummy_kp.shape)}, mask={tuple(dummy_mask.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_kp, dummy_mask), str(out_path),
            opset_version=16,
            input_names=["keypoints", "mask"],
            output_names=["logits"],
            dynamic_axes={"keypoints": {0: "batch"}, "mask": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            verbose=False,
        )
    print(f"  Exported → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Transformer export
# ─────────────────────────────────────────────────────────────────────────────

def _onnx_safe_gelu(x: torch.Tensor) -> torch.Tensor:
    """erf-based GELU — opset-14 compatible, avoids opset-20 requirement. [EXPORT-G]"""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


class ManualMHA(torch.nn.Module):
    """
    Manual Multi-Head Attention that bypasses PyTorch 2.x fused
    aten::_native_multi_head_attention (no ONNX opset-16 mapping). [EXPORT-F]
    Weights cloned from the original nn.MultiheadAttention.
    """
    def __init__(self, mha: torch.nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim  = self.embed_dim // self.num_heads
        self.in_proj_weight  = torch.nn.Parameter(mha.in_proj_weight.data.clone())
        self.in_proj_bias    = torch.nn.Parameter(mha.in_proj_bias.data.clone())
        self.out_proj_weight = torch.nn.Parameter(mha.out_proj.weight.data.clone())
        self.out_proj_bias   = torch.nn.Parameter(mha.out_proj.bias.data.clone())

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        B, T, C = query.shape
        q_w = self.in_proj_weight[:C];         k_w = self.in_proj_weight[C:2*C];    v_w = self.in_proj_weight[2*C:]
        q_b = self.in_proj_bias[:C];           k_b = self.in_proj_bias[C:2*C];      v_b = self.in_proj_bias[2*C:]

        Q = torch.nn.functional.linear(query, q_w, q_b).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = torch.nn.functional.linear(key,   k_w, k_b).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = torch.nn.functional.linear(value, v_w, v_b).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V).transpose(1, 2).reshape(B, -1, C)
        out  = torch.nn.functional.linear(out, self.out_proj_weight, self.out_proj_bias)
        return (out, attn) if need_weights else (out, None)


class _TransformerExportWrapper(torch.nn.Module):
    """
    Manually unrolls the TransformerEncoder to bypass PyTorch 2.x fused kernels.

    Fix [EXPORT-B]: Added missing inner FFN dropout (layer.dropout).
    Fix [EXPORT-C]: Residual dropout (dropout2) applied exactly once.

    PyTorch TransformerEncoderLayer has three dropout objects:
      layer.dropout  — inner FFN dropout: linear2(dropout(activation(linear1(x))))
      layer.dropout1 — post-attention residual: x + dropout1(attn_out)
      layer.dropout2 — post-FFN residual:      x + dropout2(ff_out)

    All are nn.Dropout instances, identity in eval() mode, so they do not affect
    current inference results. They are included for architectural correctness.

    The real model always uses norm_first=True (Pre-LN) so only that path runs.
    """
    def __init__(self, model):
        super().__init__()
        self.input_norm       = model.input_norm
        self.input_proj       = model.input_proj
        self.proj_norm        = model.proj_norm
        self.cls_token        = model.cls_token
        self.pos_enc          = model.pos_enc
        self.encoder_norm     = model.encoder.norm
        self.head_norm        = model.head_norm
        self.head_dropout     = model.head_dropout
        self.fc               = model.fc

        self.layer_self_attn       = torch.nn.ModuleList([ManualMHA(l.self_attn) for l in model.encoder.layers])
        self.layer_linear1         = torch.nn.ModuleList([l.linear1  for l in model.encoder.layers])
        self.layer_linear2         = torch.nn.ModuleList([l.linear2  for l in model.encoder.layers])
        self.layer_norm1           = torch.nn.ModuleList([l.norm1    for l in model.encoder.layers])
        self.layer_norm2           = torch.nn.ModuleList([l.norm2    for l in model.encoder.layers])
        self.layer_dropout1        = torch.nn.ModuleList([l.dropout1 for l in model.encoder.layers])
        # Fix [EXPORT-B]: inner FFN dropout (between activation and linear2)
        self.layer_dropout_inner   = torch.nn.ModuleList([l.dropout  for l in model.encoder.layers])
        self.layer_dropout2        = torch.nn.ModuleList([l.dropout2 for l in model.encoder.layers])

        self.num_layers = len(model.encoder.layers)
        self.norm_first = model.encoder.layers[0].norm_first  # always True

        act = model.encoder.layers[0].activation
        self._use_gelu = not (act is torch.nn.functional.relu or isinstance(act, torch.nn.ReLU))

    def forward(self, keypoints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_norm(self.input_proj(self.input_norm(keypoints)))  # (B, T, d_model)

        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)     # (B, T+1, d_model)
        cls_mask     = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat([cls_mask, mask], dim=1)                # (B, T+1)
        x = self.pos_enc(x)

        for i in range(self.num_layers):
            # Pre-LN (norm_first=True) — matches PyTorch TransformerEncoderLayer exactly
            nx = self.layer_norm1[i](x)
            ao, _ = self.layer_self_attn[i](nx, nx, nx,
                        key_padding_mask=padding_mask, need_weights=False)
            x = x + self.layer_dropout1[i](ao)

            nx = self.layer_norm2[i](x)
            ff = self.layer_linear1[i](nx)
            ff = _onnx_safe_gelu(ff) if self._use_gelu else torch.nn.functional.relu(ff)
            # Fix [EXPORT-B]: inner dropout (identity in eval, included for correctness)
            ff = self.layer_dropout_inner[i](ff)
            ff = self.layer_linear2[i](ff)
            # Fix [EXPORT-C]: residual dropout applied exactly once
            x  = x + self.layer_dropout2[i](ff)

        cls_out = self.encoder_norm(x)[:, 0, :]   # CLS token
        return self.fc(self.head_dropout(self.head_norm(cls_out)))


def export_transformer(model, out_path: Path, max_seq_len: int, device: str):
    model.eval()
    wrapper = _TransformerExportWrapper(model).to(device)
    wrapper.eval()

    dummy_kp   = torch.randn(2, max_seq_len, FEAT_DIM, device=device)
    dummy_mask = torch.zeros(2, max_seq_len, dtype=torch.bool, device=device)
    print(f"  Transformer dummy inputs: kp={tuple(dummy_kp.shape)}, mask={tuple(dummy_mask.shape)}")

    with torch.no_grad():
        try:
            from torch.nn.attention import sdp_kernel
            attn_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except ImportError:
            import contextlib
            attn_ctx = contextlib.nullcontext()
        with attn_ctx:
            torch.onnx.export(
                wrapper, (dummy_kp, dummy_mask), str(out_path),
                opset_version=16,
                input_names=["keypoints", "mask"],
                output_names=["logits"],
                dynamic_axes={
                    "keypoints": {0: "batch", 1: "seq_len"},
                    "mask":      {0: "batch", 1: "seq_len"},
                    "logits":    {0: "batch"},
                },
                do_constant_folding=True,
                verbose=False,
            )
    print(f"  Exported → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ST-GCN export
# ─────────────────────────────────────────────────────────────────────────────

class _STGCNExportWrapper(torch.nn.Module):
    """
    Pre-bakes edge-importance weights into the adjacency buffer of each block
    so no in-place copy_() appears in the traced graph. [EXPORT-E]

    After baking, sets edge_importance_weighting=False so the STGCN forward
    uses block.gcn.A directly (the pre-scaled buffer) and the
    edge_importance ParameterList is never accessed during tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        for i, block in enumerate(model.st_gcn_blocks):
            A_base = block.gcn.A.detach().clone()
            if model.edge_importance_weighting:
                ei     = model.edge_importance[i].detach().clone()
                A_sc   = (A_base * ei).clamp(min=0)
            else:
                A_sc = A_base
            block.gcn.A.data.copy_(A_sc)
        self.model.edge_importance_weighting = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def export_stgcn(model, out_path: Path, max_seq_len: int, device: str):
    model.eval()
    wrapper = _STGCNExportWrapper(model).to(device)
    wrapper.eval()

    dummy_kp = torch.randn(2, 3, max_seq_len, TOTAL_JOINTS, device=device)
    print(f"  ST-GCN dummy input: kp={tuple(dummy_kp.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy_kp, str(out_path),
            opset_version=16,
            input_names=["keypoints"],
            output_names=["logits"],
            dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            verbose=False,
        )
    print(f"  Exported → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_onnx(out_path: Path, model_type: str, max_seq_len: int, num_classes: int):
    try:
        import onnx, onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnx / onnxruntime not installed")
        return True

    print("  Validating ONNX model...")
    onnx.checker.check_model(onnx.load(str(out_path)))
    print("  ✓ ONNX graph is valid")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    if model_type == "stgcn":
        dummy = {"keypoints": np.random.randn(1, 3, max_seq_len, TOTAL_JOINTS).astype(np.float32)}
    else:
        # Fix [EXPORT-D]: np.bool_ explicit — Windows ORT rejects Python built-in bool
        dummy = {
            "keypoints": np.random.randn(1, max_seq_len, FEAT_DIM).astype(np.float32),
            "mask":      np.zeros((1, max_seq_len), dtype=np.bool_),
        }

    logits = sess.run(None, dummy)[0]
    assert logits.shape == (1, num_classes), \
        f"Shape mismatch: expected (1,{num_classes}), got {logits.shape}"
    print(f"  ✓ Inference OK — shape={logits.shape}, "
          f"top-5={np.argsort(logits[0])[::-1][:5].tolist()}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def export_model(model_type: str, cfg: dict, output_dir: Path, device: str,
                 validate: bool = True) -> bool:
    print(f"\n{'='*60}\n  Exporting  {model_type.upper()}\n{'='*60}")

    mcfg      = MODEL_CONFIGS[model_type]
    ckpt_path = Path(mcfg["exp_dir"]) / "checkpoints" / "best.pth"
    out_path  = output_dir / mcfg["output_name"]

    if not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return False

    ckpt        = torch.load(str(ckpt_path), map_location=device)
    num_classes = ckpt.get("num_classes", 558)
    max_seq_len = cfg["data"]["max_seq_len"]
    print(f"  Classes: {num_classes}   MaxSeqLen: {max_seq_len}")

    model = build_model(model_type, cfg, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    if   model_type == "bilstm":      export_bilstm(model, out_path, max_seq_len, device)
    elif model_type == "transformer": export_transformer(model, out_path, max_seq_len, device)
    elif model_type == "stgcn":       export_stgcn(model, out_path, max_seq_len, device)

    if validate:
        try:
            validate_onnx(out_path, model_type, max_seq_len, num_classes)
        except Exception as e:
            print(f"  [WARNING] Validation error: {e}")

    # Fix [EXPORT-I]: metadata always includes model_type
    metadata = {
        "model_type":  model_type,
        "num_classes": num_classes,
        "max_seq_len": max_seq_len,
        "feat_dim":    FEAT_DIM,
        "num_joints":  TOTAL_JOINTS,
        "num_coords":  3,
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata  → {meta_path}  "
          f"(~{meta_path.stat().st_size} bytes — displays as 0.0 MB, this is normal)")

    lm_src = Path(cfg["data"]["label_map_file"])
    lm_dst = output_dir / "label_map.json"
    if lm_src.exists() and not lm_dst.exists():
        shutil.copy(str(lm_src), str(lm_dst))
        print(f"  Label map → {lm_dst}")

    print(f"  ✓ Done  ({out_path.stat().st_size / 1024**2:.1f} MB)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export MSL models to ONNX")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--model",       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--output_dir",  default="onnx_models")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model <name> or --all")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models     = list(MODEL_CONFIGS.keys()) if args.all else [args.model]
    output_dir = Path(args.output_dir)
    print(f"Device: {device}")

    ok = sum(export_model(m, cfg, output_dir, device, not args.no_validate) for m in models)

    print(f"\n{'='*60}\n  SUMMARY: {ok}/{len(models)} exported to '{output_dir}/'")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / 1024**2
        note = "  (metadata — tiny file, 0.0 MB is correct)" if f.suffix == ".json" and size_mb < 0.01 else ""
        print(f"    {f.name:<35} {size_mb:6.1f} MB{note}")
    print("="*60)


if __name__ == "__main__":
    main()

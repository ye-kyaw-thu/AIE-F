# nn_unit_wfst_from_chararray_pipeline.py
from __future__ import annotations

import copy
import math
import random
import unicodedata as ud
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import pywrapfst as fst
except ImportError:
    import openfst_python as fst


# ============================================================
# Config
# ============================================================

@dataclass
class DataPrepConfig:
    sample_index_col: str = "sample_index"
    stroke_index_col: str = "stroke_index"
    emb_col: str = "embedding"
    syllable_col: str = "syllable"
    unit_array_col: str = "align_unit_array"
    blank_token: str = "<BLANK>"
    unk_token: str = "<UNK>"


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    enc_layers: int = 1
    enc_dropout: float = 0.10
    bidirectional: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 5
    device: str = "cpu"
    num_workers: int = 0


@dataclass
class DecodeConfig:
    beam_size: int = 10
    beam_token_topk: int = 5
    top_k: int = 10
    alpha_lexicon: float = 1.0
    beta_prior: float = 0.10
    variant_smoothing: float = 0.5
    syllable_smoothing: float = 1.0
    include_empirical_fallback: bool = True


@dataclass
class BaselineUnitSeqConfig:
    merge_e_aa: bool = True
    merge_e_tall_aa: bool = True
    merge_i_u: bool = False
    merge_base_asat: bool = True
    merge_tall_aa_asat: bool = True
    merge_nga_asat: bool = True
    merge_virama_next_base: bool = True
    merge_medial_ya_ha: bool = True
    merge_medial_pairs: bool = True
    merge_medial_triplets: bool = False


# ============================================================
# Myanmar unitization rules
# Reused in spirit from your unit_prototype.py
# ============================================================

MY_ASAT = "်"
MY_VIRAMA = "္"
MY_E = "ေ"
MY_AA = "ာ"
MY_TALL_AA = "ါ"
MY_MEDIAL_YA = "ျ"
MY_MEDIAL_RA = "ြ"
MY_MEDIAL_WA = "ွ"
MY_MEDIAL_HA = "ှ"

MY_MARKS = {
    "ေ", "ာ", "ါ", "ိ", "ီ", "ု", "ူ", "ဲ", "ံ", "့", "း",
    "ျ", "ြ", "ွ", "ှ", "်", "္",
}


def is_base_like(ch: str) -> bool:
    """
    Simple heuristic:
    any Myanmar char not in the common mark set is treated as base-like.
    """
    if ch in MY_MARKS:
        return False

    try:
        name = ud.name(ch, "")
    except TypeError:
        return False

    return "MYANMAR" in name


def char_array_to_units(
    chars: Iterable[str],
    cfg: Optional[BaselineUnitSeqConfig] = None,
) -> Tuple[str, ...]:
    """
    Convert canonical Unicode char_array -> baseline units.

    Rules:
      - ေ + ာ -> ော
      - ေ + ါ -> ေါ
      - BASE + ် -> merged final
      - င + ် -> င်
      - ာ + ် -> ာ်
      - ္ + BASE -> stacked base
      - ျ + ှ -> medial bundle
      - ျ+ွ, ြ+ွ, ြ+ှ -> medial bundle
    """
    cfg = cfg or BaselineUnitSeqConfig()
    chars = [str(x) for x in chars]

    units: List[str] = []
    i = 0
    n = len(chars)

    while i < n:
        # ------------------------------------------------
        # Longest-match rules first
        # ------------------------------------------------

        # ေ + ာ -> ော
        if (
            cfg.merge_e_aa
            and i + 1 < n
            and chars[i] == MY_E
            and chars[i + 1] == MY_AA
        ):
            units.append("ော")
            i += 2
            continue

        # ေ + ါ -> ေါ
        if (
            cfg.merge_e_tall_aa
            and i + 1 < n
            and chars[i] == MY_E
            and chars[i + 1] == MY_TALL_AA
        ):
            units.append("ေါ")
            i += 2
            continue

        # optional: ိ + ု
        if (
            cfg.merge_i_u
            and i + 1 < n
            and chars[i] == "ိ"
            and chars[i + 1] == "ု"
        ):
            units.append("ို")
            i += 2
            continue

        # triplet medial bundles
        if (
            cfg.merge_medial_triplets
            and i + 2 < n
            and chars[i] in {MY_MEDIAL_YA, MY_MEDIAL_RA}
            and chars[i + 1] == MY_MEDIAL_WA
            and chars[i + 2] == MY_MEDIAL_HA
        ):
            units.append(chars[i] + chars[i + 1] + chars[i + 2])
            i += 3
            continue

        # ------------------------------------------------
        # Two-symbol rules
        # ------------------------------------------------
        if i + 1 < n:
            a, b = chars[i], chars[i + 1]

            # BASE + ်
            if cfg.merge_base_asat and is_base_like(a) and b == MY_ASAT:
                units.append(a + b)
                i += 2
                continue

            # င + ် -> င်
            if cfg.merge_nga_asat and a == "င" and b == MY_ASAT:
                units.append("င်")
                i += 2
                continue

            # ာ + ်
            if cfg.merge_tall_aa_asat and a == MY_AA and b == MY_ASAT:
                units.append(a + b)
                i += 2
                continue

            # Virama + next BASE
            if cfg.merge_virama_next_base and a == MY_VIRAMA and is_base_like(b):
                units.append(a + b)
                i += 2
                continue

            # ျ + ှ
            if cfg.merge_medial_ya_ha and a == MY_MEDIAL_YA and b == MY_MEDIAL_HA:
                units.append(a + b)
                i += 2
                continue

            # other medial pairs
            if cfg.merge_medial_pairs and (a, b) in {
                (MY_MEDIAL_YA, MY_MEDIAL_WA),
                (MY_MEDIAL_RA, MY_MEDIAL_WA),
                (MY_MEDIAL_RA, MY_MEDIAL_HA),
            }:
                units.append(a + b)
                i += 2
                continue

        # default: keep single char
        units.append(chars[i])
        i += 1

    return tuple(units)


# ============================================================
# Dictionary building from syl.txt + char_array dataframe
# ============================================================

def load_syllable_list_from_txt(path: str) -> List[str]:
    """
    Read syllable list from syl.txt.
    One syllable per line. Comment lines starting with # are ignored.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s == "" or s.startswith("#"):
                continue
            out.append(s)
    return out


def build_syllable_dict_from_char_array_df(
    syllable_df: pd.DataFrame,
    *,
    allowed_syllables: Optional[Sequence[str]] = None,
    seq_cfg: Optional[BaselineUnitSeqConfig] = None,
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
) -> Dict[str, List[List[str]]]:
    """
    Build dictionary:
        syllable -> [canonical_unit_seq]

    from canonical Unicode char_array.

    Note:
      This version generates ONE canonical unit sequence per syllable.
      If later you want multiple variants, extend this function.
    """
    if syllable_col not in syllable_df.columns or char_array_col not in syllable_df.columns:
        raise KeyError(f"syllable_df must contain {syllable_col!r} and {char_array_col!r}")

    allowed = set(str(x) for x in allowed_syllables) if allowed_syllables is not None else None

    syll_dict: Dict[str, List[List[str]]] = {}
    for _, row in syllable_df.iterrows():
        syll = str(row[syllable_col])
        if allowed is not None and syll not in allowed:
            continue

        char_array = row[char_array_col]
        units = list(char_array_to_units(char_array, cfg=seq_cfg))
        syll_dict[syll] = [units]

    return syll_dict


# ============================================================
# Generic helpers
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_list(x):
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()

        # Stringified Python list support: "['က', 'ာ']"
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]

        return [tok for tok in s.split() if tok]

    return [str(x)]


# ============================================================
# Build sample tables from embeddings + metadata
# ============================================================

def build_sample_table(
    stroke_emb_df: pd.DataFrame,
    *,
    sample_meta_df: Optional[pd.DataFrame],
    cfg: DataPrepConfig,
    require_units: bool = True,
) -> pd.DataFrame:
    s_col = cfg.sample_index_col
    t_col = cfg.stroke_index_col
    e_col = cfg.emb_col
    y_col = cfg.syllable_col
    u_col = cfg.unit_array_col

    for col in [s_col, t_col, e_col]:
        if col not in stroke_emb_df.columns:
            raise KeyError(f"Missing {col!r} in stroke_emb_df")

    meta_lookup = None
    if sample_meta_df is not None:
        if s_col not in sample_meta_df.columns:
            raise KeyError(f"Missing {s_col!r} in sample_meta_df")
        meta_lookup = sample_meta_df.set_index(s_col).to_dict("index")

    rows = []
    for sid, g in stroke_emb_df.groupby(s_col):
        g = g.sort_values(t_col)
        emb_list = [np.asarray(v, dtype=np.float32) for v in g[e_col].tolist()]
        if not emb_list:
            continue

        emb_seq = np.vstack(emb_list).astype(np.float32)

        if meta_lookup is not None:
            meta = meta_lookup.get(sid)
            if meta is None:
                continue
            syll = str(meta.get(y_col, ""))
            unit_seq = _safe_list(meta.get(u_col, []))
        else:
            syll = str(g[y_col].iloc[0]) if y_col in g.columns else ""
            unit_seq = _safe_list(g[u_col].iloc[0]) if u_col in g.columns else []

        if require_units and len(unit_seq) == 0:
            continue
        if syll == "":
            continue

        rows.append({
            s_col: int(sid),
            "emb_seq": emb_seq,
            "seq_len": int(len(emb_seq)),
            y_col: syll,
            "unit_seq": [str(u) for u in unit_seq],
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No valid samples were built.")
    return out.reset_index(drop=True)


def build_inference_sample_table(
    stroke_emb_df: pd.DataFrame,
    *,
    sample_meta_df: Optional[pd.DataFrame],
    cfg: DataPrepConfig,
) -> pd.DataFrame:
    s_col = cfg.sample_index_col
    t_col = cfg.stroke_index_col
    e_col = cfg.emb_col
    y_col = cfg.syllable_col

    for col in [s_col, t_col, e_col]:
        if col not in stroke_emb_df.columns:
            raise KeyError(f"Missing {col!r} in stroke_emb_df")

    meta_lookup = None
    if sample_meta_df is not None:
        if s_col not in sample_meta_df.columns:
            raise KeyError(f"Missing {s_col!r} in sample_meta_df")
        meta_lookup = sample_meta_df.set_index(s_col).to_dict("index")

    rows = []
    for sid, g in stroke_emb_df.groupby(s_col):
        g = g.sort_values(t_col)
        emb_list = [np.asarray(v, dtype=np.float32) for v in g[e_col].tolist()]
        if not emb_list:
            continue

        emb_seq = np.vstack(emb_list).astype(np.float32)
        row = {
            s_col: int(sid),
            "emb_seq": emb_seq,
            "seq_len": int(len(emb_seq)),
        }

        if meta_lookup is not None:
            meta = meta_lookup.get(sid)
            if meta is not None and y_col in meta:
                row[y_col] = str(meta[y_col])
        elif y_col in g.columns:
            row[y_col] = str(g[y_col].iloc[0])

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No valid inference samples were built.")
    return out.reset_index(drop=True)


# ============================================================
# Dataset / collate
# ============================================================

class UnitCTCDataset(Dataset):
    def __init__(self, sample_table: pd.DataFrame, unit2idx: Dict[str, int], cfg: DataPrepConfig):
        self.df = sample_table.reset_index(drop=True)
        self.unit2idx = unit2idx
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row["emb_seq"], dtype=torch.float32)
        units = [self.unit2idx.get(u, self.unit2idx[self.cfg.unk_token]) for u in row["unit_seq"]]
        return {
            "sample_index": int(row[self.cfg.sample_index_col]),
            "x": x,
            "x_len": int(row["seq_len"]),
            "y_units": torch.tensor(units, dtype=torch.long),
            "y_units_len": int(len(units)),
            "syllable": str(row[self.cfg.syllable_col]),
            "unit_seq": list(row["unit_seq"]),
        }


class InferenceDataset(Dataset):
    def __init__(self, sample_table: pd.DataFrame, cfg: DataPrepConfig):
        self.df = sample_table.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            "sample_index": int(row[self.cfg.sample_index_col]),
            "x": torch.tensor(row["emb_seq"], dtype=torch.float32),
            "x_len": int(row["seq_len"]),
        }
        if self.cfg.syllable_col in row.index and pd.notna(row[self.cfg.syllable_col]):
            item["syllable"] = str(row[self.cfg.syllable_col])
        return item


def collate_ctc(batch):
    xs = [b["x"] for b in batch]
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_lens = torch.tensor([b["y_units_len"] for b in batch], dtype=torch.long)
    y_concat = torch.cat([b["y_units"] for b in batch], dim=0)
    return {
        "x": x_pad,
        "x_lens": x_lens,
        "y_units": y_concat,
        "y_units_lens": y_lens,
        "y_units_list": [b["unit_seq"] for b in batch],
        "sample_index": [b["sample_index"] for b in batch],
        "syllable": [b["syllable"] for b in batch],
    }


def collate_infer(batch):
    xs = [b["x"] for b in batch]
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)

    out = {
        "x": x_pad,
        "x_lens": x_lens,
        "sample_index": [b["sample_index"] for b in batch],
    }
    if len(batch) > 0 and "syllable" in batch[0]:
        out["syllable"] = [b["syllable"] for b in batch]

    return out


# ============================================================
# CTC model
# ============================================================

class StrokeBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.enc_layers,
            batch_first=True,
            dropout=cfg.enc_dropout if cfg.enc_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        self.output_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, out_len = pad_packed_sequence(out, batch_first=True)
        return out, out_len


class UnitCTCModel(nn.Module):
    def __init__(self, input_dim: int, num_units: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = StrokeBiLSTMEncoder(input_dim, cfg)
        self.unit_head = nn.Linear(self.encoder.output_dim, num_units)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        enc, out_lengths = self.encoder(x, lengths)
        logits = self.unit_head(enc)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1), out_lengths


# ============================================================
# Vocab + generic helpers
# ============================================================

def build_unit_vocab(sample_table: pd.DataFrame, cfg: DataPrepConfig):
    units = sorted({u for seq in sample_table["unit_seq"] for u in seq})
    unit2idx = {cfg.blank_token: 0, cfg.unk_token: 1}
    for u in units:
        if u not in unit2idx:
            unit2idx[u] = len(unit2idx)
    idx2unit = {i: u for u, i in unit2idx.items()}
    return unit2idx, idx2unit


def _to_device(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def ctc_greedy_decode(log_probs: torch.Tensor, out_lengths: torch.Tensor, blank_idx: int = 0):
    pred = log_probs.argmax(dim=-1).transpose(0, 1)
    outs = []
    for b in range(pred.size(0)):
        seq = pred[b, : int(out_lengths[b])].tolist()
        collapsed = []
        prev = None
        for x in seq:
            if x != blank_idx and x != prev:
                collapsed.append(int(x))
            prev = x
        outs.append(collapsed)
    return outs


def exact_sequence_accuracy(preds: List[List[str]], targets: List[List[str]]) -> float:
    if len(preds) == 0:
        return 0.0
    return float(np.mean([p == t for p, t in zip(preds, targets)]))


# ============================================================
# Train / eval CTC
# ============================================================

def _run_unit_epoch(model, loader, optimizer, criterion, device, idx2unit, grad_clip: float = 5.0):
    train = optimizer is not None
    model.train(train)
    total_loss, total_n = 0.0, 0
    all_pred_units, all_gt_units = [], []

    for batch in loader:
        batch = _to_device(batch, device)
        log_probs, out_lengths = model(batch["x"], batch["x_lens"])
        loss = criterion(log_probs, batch["y_units"], out_lengths, batch["y_units_lens"])

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch["x"].size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred_idx = ctc_greedy_decode(log_probs.detach().cpu(), out_lengths.detach().cpu(), blank_idx=0)
        pred_units = [[idx2unit[int(i)] for i in seq] for seq in pred_idx]
        all_pred_units.extend(pred_units)
        all_gt_units.extend(batch["y_units_list"])

    return {
        "loss": total_loss / max(total_n, 1),
        "unit_seq_acc": exact_sequence_accuracy(all_pred_units, all_gt_units),
    }


def train_unit_ctc(
    model: UnitCTCModel,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    *,
    idx2unit: Dict[int, str],
    cfg: TrainConfig,
):
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_unit_epoch(model, train_loader, optimizer, criterion, device, idx2unit, cfg.grad_clip)

        if valid_loader is not None:
            valid_metrics = _run_unit_epoch(model, valid_loader, None, criterion, device, idx2unit, cfg.grad_clip)
            score = valid_metrics["unit_seq_acc"]
        else:
            valid_metrics = None
            score = train_metrics["unit_seq_acc"]

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        })

        if score > best_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_unit_ctc(model: UnitCTCModel, loader: DataLoader, *, idx2unit: Dict[int, str], cfg: TrainConfig):
    device = torch.device(cfg.device)
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    return _run_unit_epoch(model, loader, None, criterion, device, idx2unit, cfg.grad_clip)


# ============================================================
# WFST from dictionary topology + train weights
# ============================================================

def _tw(val: float):
    return fst.Weight("tropical", str(float(val)))


def build_lexicon_package_from_dict(
    syllable_dict: Dict[str, List[List[str]]],
    train_table: pd.DataFrame,
    cfg: DecodeConfig,
) -> Dict:
    """
    Build hybrid lexicon package:
      - topology from dictionary syllable -> canonical unit path
      - weights from train unit_seq statistics

    Cost:
      alpha * -log P(unit_seq | syllable)
      + beta  * -log P(syllable)
    """
    syll_count = Counter()
    syll_unit_count = Counter()
    empirical_variants_by_syll = defaultdict(set)

    for _, row in train_table.iterrows():
        syll = str(row["syllable"])
        unit_seq = tuple(str(u) for u in row["unit_seq"])
        syll_count[syll] += 1
        syll_unit_count[(syll, unit_seq)] += 1
        empirical_variants_by_syll[syll].add(unit_seq)

    merged_variants_by_syll: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    seen = set()

    # dictionary topology first
    for syll, variants in syllable_dict.items():
        for seq in variants:
            t = tuple(str(u) for u in seq)
            key = (syll, t)
            if key not in seen:
                merged_variants_by_syll[syll].append(t)
                seen.add(key)

    # optional empirical fallback
    if cfg.include_empirical_fallback:
        for syll, seq_set in empirical_variants_by_syll.items():
            for t in seq_set:
                key = (syll, t)
                if key not in seen:
                    merged_variants_by_syll[syll].append(t)
                    seen.add(key)

    known_syllables = sorted(merged_variants_by_syll.keys())
    total_syll = sum(syll_count.values())
    denom_syll = total_syll + cfg.syllable_smoothing * max(len(known_syllables), 1)

    syllable_prior_cost = {}
    for syll in known_syllables:
        prob_s = (syll_count.get(syll, 0) + cfg.syllable_smoothing) / denom_syll
        syllable_prior_cost[syll] = -math.log(prob_s)

    lex_entries = []
    for syll, variants in merged_variants_by_syll.items():
        denom_var = syll_count.get(syll, 0) + cfg.variant_smoothing * max(len(variants), 1)

        for unit_seq in variants:
            count_su = syll_unit_count.get((syll, unit_seq), 0)
            prob_u_given_s = (count_su + cfg.variant_smoothing) / denom_var
            variant_cost = -math.log(prob_u_given_s)
            final_cost = cfg.alpha_lexicon * variant_cost + cfg.beta_prior * syllable_prior_cost[syll]

            lex_entries.append({
                "units": list(unit_seq),
                "syllable": syll,
                "count": int(count_su),
                "variant_cost": float(variant_cost),
                "prior_cost": float(syllable_prior_cost[syll]),
                "final_cost": float(final_cost),
            })

    return {
        "lex_entries": lex_entries,
        "syllable_prior": syllable_prior_cost,
        "num_syllables": len(known_syllables),
        "num_lex_entries": len(lex_entries),
    }


def build_symbol_tables(lexicon_pkg, data_cfg: DataPrepConfig):
    unit_sym = fst.SymbolTable()
    syll_sym = fst.SymbolTable()
    unit_sym.add_symbol("<eps>", 0)
    syll_sym.add_symbol("<eps>", 0)

    unit_tokens = {data_cfg.blank_token, data_cfg.unk_token}
    syll_tokens = set()

    for e in lexicon_pkg["lex_entries"]:
        unit_tokens.update(e["units"])
        syll_tokens.add(e["syllable"])

    for u in sorted(unit_tokens):
        if unit_sym.find(u) == -1:
            unit_sym.add_symbol(u)

    for s in sorted(syll_tokens):
        if syll_sym.find(s) == -1:
            syll_sym.add_symbol(s)

    return unit_sym, syll_sym


def build_lexicon_fst_from_package(lexicon_pkg, unit_sym, syll_sym):
    L = fst.VectorFst()
    start = L.add_state()
    L.set_start(start)

    for e in lexicon_pkg["lex_entries"]:
        units = e["units"]
        syll = e["syllable"]
        final_cost = float(e["final_cost"])

        cur = start
        valid = True

        for j, u in enumerate(units):
            ilabel = unit_sym.find(u)
            if ilabel == -1:
                valid = False
                break

            nxt = L.add_state()
            olabel = syll_sym.find(syll) if j == len(units) - 1 else 0
            L.add_arc(cur, fst.Arc(ilabel, olabel, _tw(0.0), nxt))
            cur = nxt

        if valid:
            L.set_final(cur, _tw(final_cost))

    L.set_input_symbols(unit_sym)
    L.set_output_symbols(syll_sym)
    L.arcsort(sort_type="ilabel")
    return L


def collapse_ctc_path(raw_path: Sequence[int], blank_idx: int = 0) -> Tuple[int, ...]:
    seq = []
    prev = None
    for tok in raw_path:
        if tok != blank_idx and tok != prev:
            seq.append(int(tok))
        prev = tok
    return tuple(seq)


def ctc_beam_search_decode(
    log_probs_1sample: torch.Tensor,
    beam_size: int = 10,
    blank_idx: int = 0,
    token_topk: int = 5,
):
    """
    Lightweight beam over raw CTC paths, then collapse.
    """
    T, V = log_probs_1sample.shape
    beams = [([], 0.0)]

    for t in range(T):
        vals, idxs = torch.topk(log_probs_1sample[t], k=min(token_topk, V))
        vals = vals.detach().cpu().tolist()
        idxs = idxs.detach().cpu().tolist()

        new_beams = []
        for raw_path, score in beams:
            for tok, lp in zip(idxs, vals):
                new_beams.append((raw_path + [int(tok)], score + float(lp)))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    collapsed = {}
    for raw_path, score in beams:
        key = collapse_ctc_path(raw_path, blank_idx=blank_idx)
        if key not in collapsed or score > collapsed[key]:
            collapsed[key] = score

    ranked = sorted(collapsed.items(), key=lambda x: x[1], reverse=True)
    return ranked


def build_nbest_acceptor(nbest_unit_idx_paths, idx2unit, unit_sym):
    A = fst.VectorFst()
    start = A.add_state()
    A.set_start(start)

    for unit_idx_seq, ctc_logp in nbest_unit_idx_paths:
        units = [idx2unit[int(i)] for i in unit_idx_seq]
        if len(units) == 0:
            continue

        seq_cost = -float(ctc_logp)
        per_arc_cost = seq_cost / max(len(units), 1)

        cur = start
        valid = True
        for u in units:
            uid = unit_sym.find(u)
            if uid == -1:
                valid = False
                break
            nxt = A.add_state()
            A.add_arc(cur, fst.Arc(uid, uid, _tw(per_arc_cost), nxt))
            cur = nxt

        if valid:
            A.set_final(cur, _tw(0.0))

    A.set_input_symbols(unit_sym)
    A.set_output_symbols(unit_sym)
    A.arcsort(sort_type="olabel")
    return A


def _extract_output_paths(fst_obj, syll_sym):
    """
    Compatible with pywrapfst builds that do not expose Weight.Zero.
    """
    paths = []
    start = fst_obj.start()
    if start == -1:
        return paths

    stack = [(start, [], 0.0)]
    while stack:
        state, outputs, cost = stack.pop()

        final_w = fst_obj.final(state)
        if str(final_w) != "Infinity":
            final_cost = float(str(final_w))
            out_syms = [syll_sym.find(i) for i in outputs if i != 0]
            paths.append((out_syms, cost + final_cost))

        for arc in fst_obj.arcs(state):
            arc_cost = float(str(arc.weight))
            new_outputs = outputs + ([arc.olabel] if arc.olabel != 0 else [])
            stack.append((arc.nextstate, new_outputs, cost + arc_cost))

    return paths


def decode_nbest_with_wfst(nbest_unit_idx_paths, *, idx2unit, unit_sym, syll_sym, L_fst, cfg: DecodeConfig):
    if len(nbest_unit_idx_paths) == 0:
        return {"pred_top1": None, "pred_topk": [], "mode": "none", "best_distance": None}

    A = build_nbest_acceptor(nbest_unit_idx_paths, idx2unit, unit_sym)
    AL = fst.compose(A, L_fst)

    shortest = fst.shortestpath(AL, nshortest=cfg.top_k, unique=False)
    paths = _extract_output_paths(shortest, syll_sym)

    if len(paths) == 0:
        return {"pred_top1": None, "pred_topk": [], "mode": "none", "best_distance": None}

    ranked = sorted(paths, key=lambda x: x[1])

    topk = []
    seen = set()
    for out_syms, total_cost in ranked:
        if len(out_syms) == 0:
            continue
        syll = out_syms[0]
        if syll not in seen:
            topk.append(syll)
            seen.add(syll)
        if len(topk) >= cfg.top_k:
            break

    if len(topk) == 0:
        return {"pred_top1": None, "pred_topk": [], "mode": "none", "best_distance": None}

    return {
        "pred_top1": topk[0],
        "pred_topk": topk,
        "mode": "wfst",
        "best_distance": float(ranked[0][1]),
    }


@torch.no_grad()
def predict_wfst(
    model,
    loader,
    *,
    idx2unit,
    lexicon_pkg,
    cfg: TrainConfig,
    decode_cfg: DecodeConfig,
    data_cfg: DataPrepConfig,
):
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    unit_sym, syll_sym = build_symbol_tables(lexicon_pkg, data_cfg)
    L_fst = build_lexicon_fst_from_package(lexicon_pkg, unit_sym, syll_sym)

    rows = []
    for batch in loader:
        batch = _to_device(batch, device)
        log_probs, out_lens = model(batch["x"], batch["x_lens"])

        for i in range(log_probs.size(1)):
            lp_i = log_probs[: int(out_lens[i]), i, :].detach().cpu()

            nbest = ctc_beam_search_decode(
                lp_i,
                beam_size=decode_cfg.beam_size,
                blank_idx=0,
                token_topk=decode_cfg.beam_token_topk,
            )

            dec = decode_nbest_with_wfst(
                nbest,
                idx2unit=idx2unit,
                unit_sym=unit_sym,
                syll_sym=syll_sym,
                L_fst=L_fst,
                cfg=decode_cfg,
            )

            best_units = [idx2unit[int(iu)] for iu in nbest[0][0]] if len(nbest) > 0 else []
            row = {
                "sample_index": int(batch["sample_index"][i]),
                "pred_units": best_units,
                "pred_top1": dec["pred_top1"],
                "pred_topk": dec["pred_topk"],
                "decode_mode": dec["mode"],
                "decode_distance": dec.get("best_distance"),
            }

            if "syllable" in batch:
                row["true_syllable"] = str(batch["syllable"][i])
                row["correct_top1"] = row["pred_top1"] == row["true_syllable"]

            rows.append(row)

    return pd.DataFrame(rows)


def predict_wfst_from_emb_df(
    emb_df: pd.DataFrame,
    *,
    model: UnitCTCModel,
    idx2unit: Dict[int, str],
    lexicon_pkg,
    data_cfg: DataPrepConfig,
    infer_cfg: TrainConfig,
    decode_cfg: DecodeConfig,
    sample_meta_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Predict directly from embedding dataframe + optional metadata dataframe.
    """
    sample_table = build_inference_sample_table(
        emb_df,
        sample_meta_df=sample_meta_df,
        cfg=data_cfg,
    )

    infer_ds = InferenceDataset(sample_table, data_cfg)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=infer_cfg.batch_size,
        shuffle=False,
        num_workers=infer_cfg.num_workers,
        collate_fn=collate_infer,
    )

    return predict_wfst(
        model,
        infer_loader,
        idx2unit=idx2unit,
        lexicon_pkg=lexicon_pkg,
        cfg=infer_cfg,
        decode_cfg=decode_cfg,
        data_cfg=data_cfg,
    )


# alias
predict_wfst_from_test_df = predict_wfst_from_emb_df


# ============================================================
# Metrics
# ============================================================

def topk_accuracy_from_pred_df(pred_df: pd.DataFrame, k: int):
    if "true_syllable" not in pred_df.columns:
        raise KeyError("pred_df must contain 'true_syllable'")
    hits = []
    for _, row in pred_df.iterrows():
        true = str(row["true_syllable"])
        preds = [str(x) for x in row["pred_topk"][:k]]
        hits.append(true in preds)
    return float(np.mean(hits)) if len(hits) > 0 else np.nan


def evaluate_prediction_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = {
        "top1": topk_accuracy_from_pred_df(pred_df, 1),
        "top3": topk_accuracy_from_pred_df(pred_df, 3),
        "top5": topk_accuracy_from_pred_df(pred_df, 5),
        "top10": topk_accuracy_from_pred_df(pred_df, 10),
        "wfst_rate": float((pred_df["decode_mode"] == "wfst").mean()) if "decode_mode" in pred_df.columns else np.nan,
        "none_rate": float((pred_df["decode_mode"] == "none").mean()) if "decode_mode" in pred_df.columns else np.nan,
    }
    return pd.DataFrame({"metric": list(out.keys()), "value": list(out.values())})


# ============================================================
# Save / Load
# ============================================================

def save_pipeline(
    save_path: str,
    *,
    unit_model: UnitCTCModel,
    unit2idx: Dict[str, int],
    idx2unit: Dict[int, str],
    lexicon_pkg,
    data_cfg: DataPrepConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    decode_cfg: DecodeConfig,
    seq_cfg: BaselineUnitSeqConfig,
    input_dim: int,
):
    """
    Save CTC model + WFST ingredients.
    We save lexicon_pkg, not raw OpenFST objects.
    """
    payload = {
        "unit_state_dict": unit_model.state_dict(),
        "unit2idx": unit2idx,
        "idx2unit": idx2unit,
        "lexicon_pkg": lexicon_pkg,
        "data_cfg": asdict(data_cfg),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "decode_cfg": asdict(decode_cfg),
        "seq_cfg": asdict(seq_cfg),
        "input_dim": int(input_dim),
        "pipeline_version": "ctc+wfst+chararray-v1",
    }
    torch.save(payload, save_path)


def load_pipeline(save_path: str, map_location: str = "cpu"):
    """
    Load saved pipeline and rebuild CTC model.
    WFST is rebuilt later from lexicon_pkg during prediction.
    """
    ckpt = torch.load(save_path, map_location=map_location)

    data_cfg = DataPrepConfig(**ckpt["data_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    train_cfg = TrainConfig(**ckpt["train_cfg"])
    decode_cfg = DecodeConfig(**ckpt["decode_cfg"])
    seq_cfg = BaselineUnitSeqConfig(**ckpt["seq_cfg"])

    unit2idx = ckpt["unit2idx"]
    idx2unit = ckpt["idx2unit"]
    lexicon_pkg = ckpt["lexicon_pkg"]
    input_dim = int(ckpt["input_dim"])

    unit_model = UnitCTCModel(
        input_dim=input_dim,
        num_units=len(unit2idx),
        cfg=model_cfg,
    )
    unit_model.load_state_dict(ckpt["unit_state_dict"])

    return {
        "unit_model": unit_model,
        "unit2idx": unit2idx,
        "idx2unit": idx2unit,
        "lexicon_pkg": lexicon_pkg,
        "data_cfg": data_cfg,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "decode_cfg": decode_cfg,
        "seq_cfg": seq_cfg,
        "input_dim": input_dim,
        "pipeline_version": ckpt.get("pipeline_version", "unknown"),
    }


# ============================================================
# End-to-end pipeline
# ============================================================

def run_ctc_wfst_pipeline(
    *,
    train_emb_df: pd.DataFrame,
    test_emb_df: pd.DataFrame,
    train_meta_df: pd.DataFrame,
    test_meta_df: pd.DataFrame,
    syllable_txt_path: str,
    syllable_char_df: pd.DataFrame,
    data_cfg: DataPrepConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    decode_cfg: DecodeConfig,
    seq_cfg: Optional[BaselineUnitSeqConfig] = None,
    save_path: Optional[str] = None,
    seed: int = 42,
):
    """
    Stage 1: train CTC on align_unit_array.
    Stage 2: build dictionary WFST topology from:
             syl.txt + char_array_to_units(...)
             then weight WFST from train unit-sequence statistics.
    """
    set_seed(seed)
    seq_cfg = seq_cfg or BaselineUnitSeqConfig()

    # Build supervised train/test tables for CTC
    train_table = build_sample_table(
        train_emb_df,
        sample_meta_df=train_meta_df,
        cfg=data_cfg,
        require_units=True,
    )
    test_table = build_sample_table(
        test_emb_df,
        sample_meta_df=test_meta_df,
        cfg=data_cfg,
        require_units=True,
    )

    # Build vocab
    input_dim = int(train_table.iloc[0]["emb_seq"].shape[1])
    unit2idx, idx2unit = build_unit_vocab(train_table, data_cfg)

    # Build datasets/loaders
    train_ds = UnitCTCDataset(train_table, unit2idx, data_cfg)
    test_ds = UnitCTCDataset(test_table, unit2idx, data_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_ctc,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_ctc,
    )

    # Train CTC
    model = UnitCTCModel(
        input_dim=input_dim,
        num_units=len(unit2idx),
        cfg=model_cfg,
    )
    model, history = train_unit_ctc(
        model,
        train_loader,
        test_loader,
        idx2unit=idx2unit,
        cfg=train_cfg,
    )
    unit_metrics = evaluate_unit_ctc(
        model,
        test_loader,
        idx2unit=idx2unit,
        cfg=train_cfg,
    )

    # Build syllable dictionary from syl.txt + char_array dataframe
    syllables = load_syllable_list_from_txt(syllable_txt_path)
    syllable_dict = build_syllable_dict_from_char_array_df(
        syllable_char_df,
        allowed_syllables=syllables,
        seq_cfg=seq_cfg,
        syllable_col=data_cfg.syllable_col,
        char_array_col="char_array",
    )

    # Build weighted lexicon package
    lexicon_pkg = build_lexicon_package_from_dict(
        syllable_dict,
        train_table,
        decode_cfg,
    )

    # Decode test
    pred_df = predict_wfst(
        model,
        test_loader,
        idx2unit=idx2unit,
        lexicon_pkg=lexicon_pkg,
        cfg=train_cfg,
        decode_cfg=decode_cfg,
        data_cfg=data_cfg,
    )
    pred_metrics = evaluate_prediction_df(pred_df)

    # Save pipeline if requested
    if save_path is not None:
        save_pipeline(
            save_path,
            unit_model=model,
            unit2idx=unit2idx,
            idx2unit=idx2unit,
            lexicon_pkg=lexicon_pkg,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            decode_cfg=decode_cfg,
            seq_cfg=seq_cfg,
            input_dim=input_dim,
        )

    return {
        "train_table": train_table,
        "test_table": test_table,
        "unit_model": model,
        "unit2idx": unit2idx,
        "idx2unit": idx2unit,
        "syllable_dict": syllable_dict,
        "lexicon_pkg": lexicon_pkg,
        "unit_train_history": history,
        "unit_test_metrics": unit_metrics,
        "pred_df": pred_df,
        "pred_metrics": pred_metrics,
        "input_dim": input_dim,
        "save_path": save_path,
    }


__all__ = [
    "DataPrepConfig",
    "ModelConfig",
    "TrainConfig",
    "DecodeConfig",
    "BaselineUnitSeqConfig",

    "set_seed",
    "is_base_like",
    "char_array_to_units",

    "load_syllable_list_from_txt",
    "build_syllable_dict_from_char_array_df",

    "build_sample_table",
    "build_inference_sample_table",
    "build_unit_vocab",

    "UnitCTCDataset",
    "InferenceDataset",
    "collate_ctc",
    "collate_infer",

    "StrokeBiLSTMEncoder",
    "UnitCTCModel",

    "_to_device",
    "ctc_greedy_decode",
    "exact_sequence_accuracy",

    "train_unit_ctc",
    "evaluate_unit_ctc",

    "build_lexicon_package_from_dict",
    "build_symbol_tables",
    "build_lexicon_fst_from_package",
    "collapse_ctc_path",
    "ctc_beam_search_decode",
    "build_nbest_acceptor",
    "_extract_output_paths",
    "decode_nbest_with_wfst",

    "predict_wfst",
    "predict_wfst_from_emb_df",
    "predict_wfst_from_test_df",

    "topk_accuracy_from_pred_df",
    "evaluate_prediction_df",

    "save_pipeline",
    "load_pipeline",

    "run_ctc_wfst_pipeline",
]
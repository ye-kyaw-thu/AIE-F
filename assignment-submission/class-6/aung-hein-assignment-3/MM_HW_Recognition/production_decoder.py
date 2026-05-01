import os
import pickle
import numpy as np
import pandas as pd

from prototype import score_query_against_all_syllables
from segmental_hmm import (
    decode_one_sample_segmental_hmm,
    build_token_lattice_from_emb_df,
)
from unit_prototype import (
    score_unit_sequence_with_grammar,
    UnitGrammarConfig,
    UnitPrototypeConfig,
)
from writing_units import infer_unicode_myanmar_role


NEG_FLOOR = -1e6


# ============================================================
# Small helpers
# ============================================================

def ensure_runtime_index_columns(
    df: pd.DataFrame,
    *,
    sample_index: int = 0,
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
):
    """
    Ensure runtime-required index columns exist.

    If sample_index is missing:
      - add a constant sample_index for the whole dataframe

    If stroke_index is missing:
      - create sequential stroke indices in current row order

    Returns a COPY.
    """
    out = df.copy()

    if sample_index_col not in out.columns:
        out[sample_index_col] = sample_index

    if stroke_index_col not in out.columns:
        out[stroke_index_col] = np.arange(len(out), dtype=int)

    return out
    
def safe_zscore(values):
    x = np.asarray(values, dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    if np.all(np.isnan(x)):
        return np.zeros_like(x)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-8:
        return np.zeros_like(x)
    return np.nan_to_num((x - mu) / sd)


def extract_segment_embeddings_keep_zeros(sample_df, segments, emb_col="embedding"):
    emb_dim = len(sample_df.iloc[0][emb_col])
    seg_embs = []
    for s, e in segments:
        if e <= s:
            seg_embs.append(np.zeros((emb_dim,), dtype=np.float32))
        else:
            X = np.vstack(sample_df.iloc[s:e][emb_col].values)
            seg_embs.append(X.mean(axis=0))
    return seg_embs


def same_first_base(label_a, label_b, syllable_to_first_base):
    a = syllable_to_first_base.get(str(label_a), None)
    b = syllable_to_first_base.get(str(label_b), None)
    return (a is not None) and (a == b)


def trust_to_unit_support(unit_trust: float) -> float:
    """
    Stronger moderate-trust curve for same-base reranking only.
    Output range: [0.0, 0.30]
    """
    t = float(unit_trust)

    if t < 0.15:
        return 0.0
    elif t < 0.35:
        return 0.18 * (t - 0.15) / 0.20
    elif t < 0.60:
        return 0.18 + 0.10 * (t - 0.35) / 0.25
    else:
        return 0.30


# ============================================================
# Candidate unit scoring
# ============================================================

def score_candidate_unit(
    sample_index,
    syllable,
    emb_df,
    item_units,
    seg_model_units,
    unit_proto_bank,
    syllable_to_units,
    unit_to_role,
    role_bigram_logprob,
    grammar_cfg,
    unit_proto_cfg,
):
    """
    Score one candidate using:
      - unit HMM score
      - unit prototype score (+ grammar)

    Also returns diagnostics for unit trust.
    """
    hmm_df = decode_one_sample_segmental_hmm(
        item_units,
        seg_model_units,
        candidate_syllables=[syllable],
        top_k=1,
    )

    if len(hmm_df) == 0:
        return {
            "unit_hmm_score": NEG_FLOOR,
            "unit_proto_score": NEG_FLOOR,
            "first_base": None,
            "zero_rate": 1.0,
            "bad_zero_role_count": 1,
        }

    unit_hmm_score = float(hmm_df.iloc[0]["score"])
    segments = hmm_df.iloc[0]["segments"]

    sample_df = (
        emb_df[emb_df["sample_index"] == sample_index]
        .sort_values("stroke_index")
        .reset_index(drop=True)
    )

    seg_embs = extract_segment_embeddings_keep_zeros(sample_df, segments)
    unit_seq = syllable_to_units.get(syllable, ())

    if len(unit_seq) == 0:
        return {
            "unit_hmm_score": unit_hmm_score,
            "unit_proto_score": NEG_FLOOR,
            "first_base": None,
            "zero_rate": 1.0,
            "bad_zero_role_count": 1,
        }

    unit_proto_score = score_unit_sequence_with_grammar(
        unit_embeddings=seg_embs,
        unit_labels=list(unit_seq),
        unit_bank=unit_proto_bank,
        proto_config=unit_proto_cfg,
        grammar_cfg=grammar_cfg,
        unit_to_role=unit_to_role,
        role_bigram_logprob=role_bigram_logprob,
    )

    seg_lens = [max(0, e - s) for s, e in segments]
    zero_rate = sum(l == 0 for l in seg_lens) / max(len(seg_lens), 1)

    roles = [unit_to_role.get(u, "OTHER") for u in unit_seq]
    CRIT = {"BASE", "E_BASE", "STACKED_BASE", "FINAL_ASAT", "TONE"}
    bad_zero = sum((l == 0 and r in CRIT) for l, r in zip(seg_lens, roles))

    first_base = None
    for u, r in zip(unit_seq, roles):
        if r in {"BASE", "E_BASE", "STACKED_BASE"}:
            first_base = u
            break

    return {
        "unit_hmm_score": unit_hmm_score,
        "unit_proto_score": unit_proto_score,
        "first_base": first_base,
        "zero_rate": zero_rate,
        "bad_zero_role_count": bad_zero,
    }


# ============================================================
# Candidate-level unit trust
# ============================================================

def compute_unit_trust(cand_df):
    unit_rank = cand_df.sort_values("unit_model_score", ascending=False).reset_index(drop=True)
    hmm_rank = cand_df.sort_values("unit_hmm_norm", ascending=False).reset_index(drop=True)
    proto_rank = cand_df.sort_values("unit_proto_norm", ascending=False).reset_index(drop=True)

    top = unit_rank.iloc[0]
    top_label = top["syllable"]

    hmm_gap = (
        hmm_rank.iloc[0]["unit_hmm_norm"] - hmm_rank.iloc[1]["unit_hmm_norm"]
        if len(hmm_rank) > 1 else 0.0
    )

    proto_gap = (
        proto_rank.iloc[0]["unit_proto_norm"] - proto_rank.iloc[1]["unit_proto_norm"]
        if len(proto_rank) > 1 else 0.0
    )

    same_base = unit_rank[
        (unit_rank["first_base"] == top["first_base"]) &
        (unit_rank["syllable"] != top_label)
    ]

    same_base_gap = (
        top["unit_model_score"] - same_base["unit_model_score"].max()
        if len(same_base) > 0 else 0.0
    )

    unit_agree = (
        hmm_rank.iloc[0]["syllable"] ==
        proto_rank.iloc[0]["syllable"]
    )

    seg_penalty = min(1.0, 0.5 * top["zero_rate"] + 0.5 * (top["bad_zero_role_count"] > 0))

    def squash(x, s):
        return np.clip(x / s, 0.0, 1.0)

    trust = (
        0.40 * squash(hmm_gap, 0.4) +
        0.20 * squash(proto_gap, 0.25) +
        0.20 * squash(same_base_gap, 0.30) +
        0.10 * (1.0 if unit_agree else -0.5) -
        0.10 * seg_penalty
    )

    trust = float(np.clip(trust, 0.0, 1.0))

    return trust, {
        "unit_trust": trust,
        "hmm_gap": hmm_gap,
        "proto_gap": proto_gap,
        "same_base_gap": same_base_gap,
        "unit_agree": unit_agree,
        "zero_rate": top["zero_rate"],
        "bad_zero_role_count": top["bad_zero_role_count"],
    }


# ============================================================
# Production decoder class
# ============================================================

class PiProductionDecoder:
    """
    Frozen production baseline:
      - syllable prototype = global expert
      - unit branch = same-base specialist
      - fixed blend = 0.60 / 0.40
      - trust-gated same-base rerank only
    """

    def __init__(self, artifacts: dict):
        self.prototype_bank = artifacts["prototype_bank"]
        self.unit_proto_bank = artifacts["unit_proto_bank"]
        self.seg_model_units = artifacts["seg_model_units"]
        self.syllable_to_units = artifacts["syllable_to_units"]
        self.unit_to_role = artifacts["unit_to_role"]
        self.role_bigram_logprob = artifacts["role_bigram_logprob"]
        self.syllable_to_first_base = artifacts["syllable_to_first_base"]
        # Rehydrate configs if they were exported as dicts
        raw_grammar_cfg = artifacts["grammar_cfg"]
        raw_unit_proto_cfg = artifacts["unit_proto_cfg"]
        
        self.grammar_cfg = (
            UnitGrammarConfig(**raw_grammar_cfg)
            if isinstance(raw_grammar_cfg, dict)
            else raw_grammar_cfg
        )
        
        self.unit_proto_cfg = (
            UnitPrototypeConfig(**raw_unit_proto_cfg)
            if isinstance(raw_unit_proto_cfg, dict)
            else raw_unit_proto_cfg
        )
        self.config = artifacts["config"]

    @classmethod
    def load(cls, export_dir: str):
        with open(os.path.join(export_dir, "frozen_decoder.pkl"), "rb") as f:
            artifacts = pickle.load(f)
        return cls(artifacts)

    def build_item_units_for_sample(
        self,
        sample_emb_df: pd.DataFrame,
        *,
        sample_index: int = 0,
        sample_index_col: str = "sample_index",
        stroke_index_col: str = "stroke_index",
        emb_col: str = "embedding",
    ):
        """
        Build item_units for ONE sample at runtime using the frozen unit tokenizer.
    
        Robust to missing sample_index / stroke_index.
        """
        if len(sample_emb_df) == 0:
            raise ValueError("sample_emb_df is empty")
    
        work_df = ensure_runtime_index_columns(
            sample_emb_df,
            sample_index=sample_index,
            sample_index_col=sample_index_col,
            stroke_index_col=stroke_index_col,
        )
    
        sid = int(work_df[sample_index_col].iloc[0])
    
        # Dummy columns for builder compatibility
        if "syllable" not in work_df.columns:
            work_df["syllable"] = "__DUMMY__"
    
        if "align_unit_array" not in work_df.columns:
            work_df["align_unit_array"] = [("__DUMMY__",)] * len(work_df)
    
        items = build_token_lattice_from_emb_df(
            work_df,
            self.seg_model_units["tokenizer"],
            emb_col=emb_col,
            sample_index_col=sample_index_col,
            stroke_index_col=stroke_index_col,
            syllable_col="syllable",
            char_array_col="align_unit_array",
        )
    
        return items[sid]

    def predict_from_emb_df(
            self,
            sample_index: int,
            emb_df: pd.DataFrame,
            item_units=None,
            *,
            return_debug: bool = False,
        ):
        """
        Predict Top-K for ONE sample.

        If item_units is not provided, it will be built automatically at runtime
        using the frozen unit tokenizer.
        """
        qdf = emb_df[emb_df["sample_index"] == sample_index]

        if item_units is None:
            sample_emb_df = (
                emb_df[emb_df["sample_index"] == sample_index]
                .sort_values("stroke_index")
                .reset_index(drop=True)
            )
            item_units = self.build_item_units_for_sample(sample_emb_df)

        proto_df = score_query_against_all_syllables(
            qdf,
            self.prototype_bank,
            return_details=False,
        )
        proto_map = dict(zip(proto_df["syllable_label"], proto_df["score"]))

        shortlist_k = self.config.get("shape_shortlist_k", 25)
        final_top_k = self.config.get("final_top_k", 10)
        unit_hmm_weight = self.config.get("unit_hmm_weight", 0.80)
        unit_proto_weight = self.config.get("unit_proto_weight", 0.20)
        base_blend_syll = self.config.get("base_blend_syll", 0.60)
        base_blend_unit = self.config.get("base_blend_unit", 0.40)

        shortlist = (
            proto_df.sort_values("score", ascending=False)["syllable_label"]
            .astype(str)
            .tolist()[:shortlist_k]
        )

        rows = []
        for syll in shortlist:
            uinfo = score_candidate_unit(
                sample_index=sample_index,
                syllable=syll,
                emb_df=emb_df,
                item_units=item_units,
                seg_model_units=self.seg_model_units,
                unit_proto_bank=self.unit_proto_bank,
                syllable_to_units=self.syllable_to_units,
                unit_to_role=self.unit_to_role,
                role_bigram_logprob=self.role_bigram_logprob,
                grammar_cfg=self.grammar_cfg,
                unit_proto_cfg=self.unit_proto_cfg,
            )
            rows.append({
                "syllable": syll,
                "syll_score": proto_map.get(syll, NEG_FLOOR),
                **uinfo,
            })

        cand_df = pd.DataFrame(rows)

        cand_df["syll_norm"] = safe_zscore(cand_df["syll_score"])
        cand_df["unit_hmm_norm"] = safe_zscore(cand_df["unit_hmm_score"])
        cand_df["unit_proto_norm"] = safe_zscore(cand_df["unit_proto_score"])

        cand_df["unit_model_score"] = (
            unit_hmm_weight * cand_df["unit_hmm_norm"] +
            unit_proto_weight * cand_df["unit_proto_norm"]
        )

        cand_df["final_score"] = (
            base_blend_syll * cand_df["syll_norm"] +
            base_blend_unit * cand_df["unit_model_score"]
        )

        trust, trust_info = compute_unit_trust(cand_df)

        syll_top = cand_df.sort_values("syll_norm", ascending=False).iloc[0]["syllable"]
        unit_top = cand_df.sort_values("unit_model_score", ascending=False).iloc[0]["syllable"]

        same_base_ok = same_first_base(syll_top, unit_top, self.syllable_to_first_base)
        unit_support = trust_to_unit_support(trust)

        if same_base_ok and unit_support > 0:
            target_base = self.syllable_to_first_base.get(syll_top, None)

            mask = (
                cand_df["syllable"].map(self.syllable_to_first_base) == target_base
            )

            same_base_w_unit = min(base_blend_unit + unit_support, 0.65)
            same_base_w_syll = 1.0 - same_base_w_unit

            cand_df.loc[mask, "final_score"] = (
                same_base_w_syll * cand_df.loc[mask, "syll_norm"] +
                same_base_w_unit * cand_df.loc[mask, "unit_model_score"]
            )

            cand_df.loc[cand_df["syllable"] == unit_top, "final_score"] += 0.02 * unit_support

        cand_df = cand_df.sort_values("final_score", ascending=False).reset_index(drop=True)

        final_top1 = str(cand_df.iloc[0]["syllable"])
        topk = [final_top1] + [str(s) for s in cand_df["syllable"] if str(s) != final_top1]
        topk = topk[:final_top_k]

        branch_info = {
            "conf_syll": float(cand_df["syll_norm"].max()),
            "conf_unit": trust,
            "w_syll": base_blend_syll,
            "w_unit": base_blend_unit,
            "unit_support": unit_support,
            "same_base_ok": same_base_ok,
            "top1_source": (
                "unit_same_base_support"
                if (same_base_ok and unit_support > 0)
                else "blend"
            ),
            "top1_syll_label": str(syll_top),
            "top1_unit_label": str(unit_top),
            **trust_info,
        }

        if return_debug:
            return topk, cand_df, branch_info
        return topk
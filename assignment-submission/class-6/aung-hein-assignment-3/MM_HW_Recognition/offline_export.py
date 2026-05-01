import os
import json
import pickle
from dataclasses import asdict, is_dataclass

from writing_units import infer_unicode_myanmar_role


def build_syllable_to_first_base(syllable_df):
    """
    syllable -> first canonical base from char_array
    """
    out = {}
    for _, row in syllable_df.iterrows():
        syll = str(row["syllable"])
        chars = list(row["char_array"])
        bases = [str(ch) for ch in chars if infer_unicode_myanmar_role(str(ch)) == "BASE"]
        out[syll] = bases[0] if len(bases) > 0 else None
    return out


def _maybe_asdict(x):
    if is_dataclass(x):
        return asdict(x)
    return x


def save_frozen_decoder_artifacts(
    export_dir,
    *,
    prototype_bank,
    unit_proto_bank,
    seg_model_units,
    syllable_df,
    syllable_to_units,
    unit_to_role,
    role_bigram_logprob,
    grammar_cfg,
    unit_proto_cfg,
    config=None,
):
    """
    Save everything needed for Pi runtime.
    Run this OFFLINE after training.
    """
    os.makedirs(export_dir, exist_ok=True)

    syllable_to_first_base = build_syllable_to_first_base(syllable_df)

    artifacts = {
        "prototype_bank": prototype_bank,
        "unit_proto_bank": unit_proto_bank,
        "seg_model_units": seg_model_units,
        "syllable_to_units": syllable_to_units,
        "unit_to_role": unit_to_role,
        "role_bigram_logprob": role_bigram_logprob,
        "syllable_to_first_base": syllable_to_first_base,
        "grammar_cfg": _maybe_asdict(grammar_cfg),
        "unit_proto_cfg": _maybe_asdict(unit_proto_cfg),
        "config": config or {
            "shape_shortlist_k": 25,
            "final_top_k": 10,
            "unit_hmm_weight": 0.80,
            "unit_proto_weight": 0.20,
            "base_blend_syll": 0.60,
            "base_blend_unit": 0.40,
        },
    }

    with open(os.path.join(export_dir, "frozen_decoder.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    with open(os.path.join(export_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_file": "frozen_decoder.pkl",
                "notes": "Production baseline decoder: syllable global expert + unit same-base specialist",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Saved frozen decoder artifacts to: {export_dir}")
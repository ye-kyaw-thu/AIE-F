"""
Evaluate MOSNet on a HuggingFace dataset's audio column
Author: Thura Aung
Date: 30th Aug 2025
"""
from pathlib import Path
import logging

import torch
import soundfile as sf
from datasets import load_dataset
from mosnet import MOSNet

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Audio Metrics ----------------- #
class AudioMetrics:
    def __init__(self, mos_weight: str):
        self.mos_model = MOSNet(device=DEVICE)
        self.mos_model.load_state_dict(torch.load(mos_weight, map_location=DEVICE))
        self.mos_model.to(DEVICE).eval()

    def mos(self, wav_path: Path) -> float:
        try:
            y, _ = sf.read(wav_path)
            y = y.mean(axis=1) if y.ndim > 1 else y
            y_tensor = torch.from_numpy(y.astype("float32")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mos, _ = self.mos_model(y_tensor)
            return float(mos.item())
        except Exception as e:
            logging.warning("MOS error: %s", e)
            return float("nan")

# ----------------- Evaluation ----------------- #
def evaluate_dataset(dataset_name: str, mos_weight: str, out_dir: str):
    logging.info("Loading dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, split="train")
    metrics = AudioMetrics(mos_weight)
    results = []

    Path(out_dir).mkdir(exist_ok=True)

    for i, item in enumerate(ds):
        try:
            audio_path = Path(out_dir) / f"audio_{i}.wav"
            # Save audio to temporary wav file
            sf.write(audio_path, item["audio"]["array"], item["audio"]["sampling_rate"])

            # Compute MOS
            mos_score = metrics.mos(audio_path)

            results.append({
                "index": i,
                "mos": mos_score
            })

            if i % 10 == 0:
                logging.info("[%d/%d] Processed", i, len(ds))
        except Exception as e:
            logging.error("Error processing index %d: %s", i, e)
            continue

    return results

# ----------------- Main ----------------- #
if __name__ == "__main__":
    mos_weight_path = "mosnet16_torch.pt"
    out_dir = "dataset_audio_temp"
    datasets_to_eval = [
        "Ko-Yin-Maung/mig-burmese-audio-transcription"    ]

    import pandas as pd

    for ds_name in datasets_to_eval:
        results = evaluate_dataset(ds_name, mos_weight_path, out_dir)
        out_csv = f"{ds_name.split('/')[-1]}_mosnet.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        logging.info("Saved MOS results to %s", out_csv)
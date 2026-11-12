'Measure selected visual cases without inferring causal explanations.'
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
import common


if __name__ == "__main__":
    destination = ROOT / "results/part1_course448"
    cases = pd.read_csv(destination / "visual_cases.csv")
    if len(cases) < 20 or not cases.image.is_unique:
        raise ValueError("At least 20 distinct visual cases are required")
    metrics = pd.read_csv(destination / "metrics.csv")
    records = []
    for case in cases.itertuples():
        predictions = pd.read_csv(common.OUTPUT_DIR / case.method / case.category / "test/predictions.csv")
        row = predictions[predictions.image == case.image].iloc[0]
        mask = common.load_mask(None if pd.isna(row.mask_path) else row.mask_path).astype(bool)
        heatmap = np.load(row.map_path)
        threshold = metrics.query("method == @case.method and category == @case.category").iloc[0].pixel_threshold
        predicted = heatmap > threshold
        records.append({"case_id": case.case_id, "image_score_margin": case.score - case.threshold,
                        "defect_pixel_fraction": float(mask.mean()),
                        "pixel_recall_at_normal_validation_threshold": float(predicted[mask].mean()) if mask.any() else None,
                        "background_pixel_fpr": float(predicted[~mask].mean()),
                        "mean_defect_score": float(heatmap[mask].mean()) if mask.any() else None,
                        "mean_background_score": float(heatmap[~mask].mean())})
    pd.DataFrame(records).to_csv(destination / "visual_case_statistics.csv", index=False)
    print(f"Verified {len(cases)} distinct visual cases; statistics saved")

'Shared normal-validation calibration and image/pixel evaluation.'

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, roc_auc_score
from skimage.measure import label as connected_components

from common import (
    IMAGE_SIZE,
    PROTOCOL,
    source_revision,
    OUTPUT_DIR,
    REQUIRED_CATEGORIES,
    ensure_output_dir,
    load_mask,
    load_rgb_image,
)


def calibrate_threshold(validation: pd.DataFrame, target_fpr: float) -> float:
    'Calibrate the image threshold from normal validation scores only.'
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    if not validation["label"].eq("normal").all():
        raise ValueError("Threshold calibration requires normal validation images only.")
    threshold = float(np.quantile(validation["score"].to_numpy(), 1 - target_fpr, method="higher"))

    return threshold


def calibrate_pixel_threshold(validation: pd.DataFrame, target_pixel_fpr: float = 0.01) -> float:
    'Calibrate the pixel threshold from normal validation map pixels only.'
    if not validation["label"].eq("normal").all():
        raise ValueError("Pixel threshold calibration requires normal validation data.")
    pixel_values = [np.load(path).astype(np.float32).ravel() for path in validation["map_path"]]
    normal_pixels = np.concatenate(pixel_values)
    return float(np.quantile(normal_pixels, 1 - target_pixel_fpr, method="higher"))


def evaluate_aupro(test_predictions, category, max_fpr=0.30, num_thresholds=200):
    'Compatibility wrapper for the shared 200-threshold, 8-connected AUPRO.'
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be greater than 0 and no greater than 1")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")
    background_score_chunks = []
    region_score_chunks = []
    minimum_score = np.inf
    maximum_score = -np.inf
    for _, row in test_predictions.iterrows():
        score_map = np.load(row["map_path"]).astype(np.float32)
        raw_mask_path = row["mask_path"]
        mask_path = None if pd.isna(raw_mask_path) else raw_mask_path
        true_mask = load_mask(mask_path)
        if score_map.shape != true_mask.shape:
            raise ValueError(f"Map/mask shape mismatch: {score_map.shape} vs {true_mask.shape}")
        minimum_score = min(minimum_score, float(score_map.min()))
        maximum_score = max(maximum_score, float(score_map.max()))
        background_score_chunks.append(score_map[true_mask == 0])
        component_labels = connected_components(true_mask.astype(bool), connectivity=2)
        for component_id in range(1, int(component_labels.max()) + 1):
            region_mask = component_labels == component_id
            region_score_chunks.append(score_map[region_mask])
    if not region_score_chunks:
        raise ValueError("AUPRO requires at least one annotated defect region")
    background_scores = np.sort(np.concatenate(background_score_chunks))
    region_score_chunks = [np.sort(values) for values in region_score_chunks]
    thresholds = np.linspace(maximum_score, minimum_score, num_thresholds)
    thresholds = np.concatenate(([np.inf], thresholds))
    fpr_points = []
    pro_points = []
    for threshold in thresholds:
        fpr = float((background_scores.size - np.searchsorted(background_scores, threshold, side="left")) / background_scores.size)
        region_overlaps = [
            float((region_scores.size - np.searchsorted(region_scores, threshold, side="left")) / region_scores.size) for region_scores in region_score_chunks
        ]
        pro = float(np.mean(region_overlaps))
        fpr_points.append(fpr)
        pro_points.append(pro)
    fpr_points = np.asarray(fpr_points, dtype=np.float64)
    pro_points = np.asarray(pro_points, dtype=np.float64)
    unique_fpr = np.unique(fpr_points)
    best_pro_at_fpr = np.asarray([
        pro_points[fpr_points == fpr].max() for fpr in unique_fpr
    ])
    inside_limit = unique_fpr < max_fpr
    curve_fpr = np.concatenate((unique_fpr[inside_limit], [max_fpr]))
    pro_at_limit = float(np.interp(max_fpr, unique_fpr, best_pro_at_fpr))
    curve_pro = np.concatenate((best_pro_at_fpr[inside_limit], [pro_at_limit]))
    aupro = float(auc(curve_fpr, curve_pro) / max_fpr)
    return {
        "test_aupro": aupro,
        "aupro_max_fpr": float(max_fpr),
        "aupro_num_thresholds": int(num_thresholds),
        "aupro_regions": int(len(region_score_chunks)),
    }


def compute_aupro(test_rows, anomaly_maps, max_fpr=0.3, bins=200):
    'Integrate normalized region overlap up to the specified background FPR limit.'
    table = test_rows.copy()

    return evaluate_aupro(table, "shared", max_fpr=max_fpr, num_thresholds=bins)["test_aupro"]


def load_maps(predictions: pd.DataFrame) -> list[np.ndarray]:
    return [np.load(path).astype(np.float32) for path in predictions["map_path"]]


def make_error_panel(
    category: str,
    method: str,
    test_rows: pd.DataFrame,
    anomaly_maps: list[np.ndarray],
    threshold: float,
    pixel_threshold: float,
    output_dir: Path,
) -> list[dict]:
    records = []
    for index, row in test_rows.reset_index(drop=True).iterrows():
        predicted_anomaly = float(row["score"]) > threshold
        true_anomaly = row["label"] == "anomaly"
        if predicted_anomaly != true_anomaly:
            records.append(
                {
                    "category": category,
                    "method": method,
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "score": float(row["score"]),
                    "threshold": threshold,
                    "error_type": "false_positive" if predicted_anomaly else "false_negative",
                    "_index": index,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return records

    selected = sorted(records, key=lambda item: abs(item["score"] - threshold), reverse=True)[:6]
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3.2 * len(selected)), squeeze=False)
    for row_index, record in enumerate(selected):
        index = record["_index"]
        test_row = test_rows.reset_index(drop=True).iloc[index]
        image = load_rgb_image(test_row["image_path"], IMAGE_SIZE)
        truth = load_mask(test_row["mask_path"], IMAGE_SIZE)
        anomaly_map = anomaly_maps[index]
        predicted_mask = anomaly_map > pixel_threshold
        panels = [
            (image, "Input"),
            (truth, "Ground truth"),
            (anomaly_map, "Anomaly map"),
            (predicted_mask, "Thresholded map"),
        ]
        for column, (panel, title) in enumerate(panels):
            axes[row_index, column].imshow(panel, cmap="gray" if column in (1, 3) else ("inferno" if column == 2 else None))
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(record["error_type"], rotation=0, labelpad=70, va="center")
    fig.suptitle(f"{category} / {method} error cases")
    fig.tight_layout()
    fig.savefig(output_dir / f"{category}_{method}_errors.png", dpi=140)
    plt.close(fig)
    return records


def evaluate_one(category: str, method: str, target_fpr: float = 0.05) -> tuple[dict, list[dict]]:
    method_dir = OUTPUT_DIR / method / category
    validation = pd.read_csv(method_dir / "validation" / "predictions.csv")
    test = pd.read_csv(method_dir / "test" / "predictions.csv")
    validation["mask_path"] = validation["mask_path"].where(validation["mask_path"].notna(), None)
    test["mask_path"] = test["mask_path"].where(test["mask_path"].notna(), None)
    threshold = calibrate_threshold(validation, target_fpr)
    pixel_threshold = calibrate_pixel_threshold(validation)
    test_maps = load_maps(test)
    y_image = (test["label"] == "anomaly").astype(np.uint8).to_numpy()
    image_scores = test["score"].to_numpy(dtype=np.float64)
    y_pixel = np.concatenate([load_mask(row.mask_path, IMAGE_SIZE).ravel() for row in test.itertuples()])
    score_pixel = np.concatenate([score_map.ravel() for score_map in test_maps])
    predicted_image = image_scores > threshold
    normal = y_image == 0
    false_positive_rate = float(np.mean(predicted_image[normal])) if np.any(normal) else float("nan")
    result = {
        "category": category,
        "method": method,
        "n_test": int(len(test)),
        "image_auroc": float(roc_auc_score(y_image, image_scores)),
        "image_ap": float(average_precision_score(y_image, image_scores)),
        "image_threshold_target_fpr": target_fpr,
        "image_threshold": threshold,
        "validation_image_fpr": float((validation["score"] > threshold).mean()),
        "image_fpr_at_threshold": false_positive_rate,
        "pixel_auroc": float(roc_auc_score(y_pixel, score_pixel)),
        "pixel_ap": float(average_precision_score(y_pixel, score_pixel)),
        "pixel_threshold_target_fpr": 0.01,
        "pixel_threshold": pixel_threshold,
        "aupro": compute_aupro(test, test_maps, max_fpr=0.3),
        "aupro_max_fpr": 0.3,
    }
    visualization_dir = ensure_output_dir("evaluation", "qualitative")
    errors = make_error_panel(category, method, test, test_maps, threshold, pixel_threshold, visualization_dir)
    return result, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=REQUIRED_CATEGORIES, default=REQUIRED_CATEGORIES)
    parser.add_argument("--methods", nargs="+", choices=("template", "autoencoder"), default=("template", "autoencoder"))
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()

    results, errors = [], []
    for method in args.methods:
        for category in args.categories:
            result, method_errors = evaluate_one(category, method, args.target_fpr)
            results.append(result)
            errors.extend(method_errors)
            print(
                f"[{method}/{category}] image AUROC={result['image_auroc']:.4f}, "
                f"pixel AUROC={result['pixel_auroc']:.4f}, AUPRO={result['aupro']:.4f}, "
                f"FPR={result['image_fpr_at_threshold']:.3f}"
            )

    output_dir = ensure_output_dir("evaluation")
    results_path = output_dir / "metrics.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    errors_path = output_dir / "error_cases.csv"
    pd.DataFrame(errors).drop(columns=["_index"], errors="ignore").to_csv(errors_path, index=False)
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "image_threshold": "normal validation quantile; predict score > threshold",
                "source_sha256": source_revision(),
                "aupro_thresholds": 200,
                "aupro_connectivity": 8,
                "all_pixels_evaluated": True,
                "pixel_threshold": "99th percentile of normal validation pixels (target pixel FPR 1%)",
                "aupro_max_fpr": 0.3,
                "image_score": "mean of top 1% anomaly-map pixels",
                "pixel_maps_resized_to": list(IMAGE_SIZE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Metrics: {results_path}")
    print(f"Error cases: {errors_path}")


if __name__ == "__main__":
    main()

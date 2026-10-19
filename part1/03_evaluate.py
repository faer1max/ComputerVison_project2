"""统一评估 Part 1 预测：校准阈值、计算指标并导出失败案例。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage.measure import label as connected_components

from common import (
    IMAGE_SIZE,
    OUTPUT_DIR,
    REQUIRED_CATEGORIES,
    ensure_output_dir,
    load_mask,
    load_rgb_image,
)


def calibrate_threshold(validation: pd.DataFrame, target_fpr: float) -> float:
    """用正常验证图的分数分位数设图像级阈值，不读取测试标签。"""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr 必须在 0 和 1 之间。")
    if not validation["label"].eq("normal").all():
        raise ValueError("阈值校准集必须只包含正常图。")
    threshold = float(np.quantile(validation["score"].to_numpy(), 1 - target_fpr, method="higher"))
    # 使用 >= 作判定时，阈值处的并列分数可能让验证 FPR 超过目标；向上挪一个浮点间隔。
    return float(np.nextafter(threshold, np.inf))


def calibrate_pixel_threshold(validation: pd.DataFrame, target_pixel_fpr: float = 0.01) -> float:
    """用正常验证热图的像素分位数设可视化/分割阈值。"""
    pixel_values = [np.load(path).astype(np.float32).ravel() for path in validation["map_path"]]
    normal_pixels = np.concatenate(pixel_values)
    return float(np.quantile(normal_pixels, 1 - target_pixel_fpr, method="higher"))


def compute_aupro(
    test_rows: pd.DataFrame,
    anomaly_maps: list[np.ndarray],
    max_fpr: float = 0.3,
    bins: int = 128,
) -> float:
    """按连通缺陷区域计算 PRO 曲线，并在指定 FPR 范围内归一化积分。"""
    negative_scores = []
    region_histograms = []
    all_scores = []
    for row, anomaly_map in zip(test_rows.itertuples(), anomaly_maps):
        ground_truth = load_mask(row.mask_path, IMAGE_SIZE).astype(bool)
        score_map = np.asarray(anomaly_map, dtype=np.float32)
        if score_map.shape != ground_truth.shape:
            raise ValueError(f"热图与掩码尺寸不一致：{score_map.shape} vs {ground_truth.shape}")
        negative_scores.append(score_map[~ground_truth])
        all_scores.append(score_map.ravel())
        components = connected_components(ground_truth, connectivity=2)
        for region_id in range(1, int(components.max()) + 1):
            region_scores = score_map[components == region_id]
            if region_scores.size:
                region_histograms.append(region_scores)

    if not region_histograms:
        return float("nan")
    negatives = np.concatenate(negative_scores)
    scores = np.concatenate(all_scores)
    low, high = float(scores.min()), float(scores.max())
    if high <= low:
        return 0.0
    edges = np.linspace(low, high + np.finfo(np.float32).eps, bins + 1)
    fpr_hist, _ = np.histogram(negatives, bins=edges)
    fpr_curve = np.cumsum(fpr_hist[::-1])[::-1] / max(1, negatives.size)
    pro_curves = []
    for region_scores in region_histograms:
        hist, _ = np.histogram(region_scores, bins=edges)
        pro_curves.append(np.cumsum(hist[::-1])[::-1] / region_scores.size)
    pro_curve = np.mean(pro_curves, axis=0)

    order = np.argsort(fpr_curve)
    x, y = fpr_curve[order], pro_curve[order]
    unique_x = np.unique(x)
    unique_y = np.asarray([np.max(y[x == point]) for point in unique_x])
    endpoint = min(max_fpr, float(unique_x[-1]))
    if endpoint <= 0:
        return 0.0
    inside = unique_x < endpoint
    x_plot = np.concatenate(([0.0], unique_x[inside], [endpoint]))
    y_plot = np.concatenate(([unique_y[0]], unique_y[inside], [np.interp(endpoint, unique_x, unique_y)]))
    return float(np.trapezoid(y_plot, x_plot) / max_fpr)


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
        predicted_anomaly = float(row["score"]) >= threshold
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
    # 每类/方法最多展示分数最偏离阈值的 6 个错误，完整错误列表仍写入 CSV。
    selected = sorted(records, key=lambda item: abs(item["score"] - threshold), reverse=True)[:6]
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3.2 * len(selected)), squeeze=False)
    for row_index, record in enumerate(selected):
        index = record["_index"]
        test_row = test_rows.reset_index(drop=True).iloc[index]
        image = load_rgb_image(test_row["image_path"], IMAGE_SIZE)
        truth = load_mask(test_row["mask_path"], IMAGE_SIZE)
        anomaly_map = anomaly_maps[index]
        predicted_mask = anomaly_map >= pixel_threshold
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
    predicted_image = image_scores >= threshold
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
                "image_threshold": "95th percentile of normal validation image scores (target FPR 5%)",
                "pixel_threshold": "99th percentile of normal validation pixels (target pixel FPR 1%)",
                "aupro_max_fpr": 0.3,
                "image_score": "mean of top 1% anomaly-map pixels",
                "pixel_maps_resized_to": list(IMAGE_SIZE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"指标：{results_path}")
    print(f"错误案例：{errors_path}")


if __name__ == "__main__":
    main()

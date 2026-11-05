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
    """用正常验证图的分数分位数设图像级阈值，不读取测试标签。"""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr 必须在 0 和 1 之间。")
    if not validation["label"].eq("normal").all():
        raise ValueError("阈值校准集必须只包含正常图。")
    threshold = float(np.quantile(validation["score"].to_numpy(), 1 - target_fpr, method="higher"))
    # 统一使用严格大于，分数相同的样本不会因并列而被误报。
    return threshold


def calibrate_pixel_threshold(validation: pd.DataFrame, target_pixel_fpr: float = 0.01) -> float:
    """用正常验证热图的像素分位数设可视化/分割阈值。"""
    if not validation["label"].eq("normal").all():
        raise ValueError("像素阈值只允许正常验证数据")
    pixel_values = [np.load(path).astype(np.float32).ravel() for path in validation["map_path"]]
    normal_pixels = np.concatenate(pixel_values)
    return float(np.quantile(normal_pixels, 1 - target_pixel_fpr, method="higher"))


def evaluate_aupro(test_predictions, category, max_fpr=0.30, num_thresholds=200):
    """计算标准区域级 PRO 曲线在 FPR 上限内的归一化面积 AUPRO。"""
    if not 0 < max_fpr <= 1:  # 积分上限必须是有效的正 FPR 比例。
        raise ValueError("max_fpr 必须大于 0 且小于等于 1")  # 参数错误时停止，避免产生无效归一化。
    if num_thresholds < 2:  # 曲线至少需要两个阈值点才能估计面积。
        raise ValueError("num_thresholds 至少为 2")  # 点数不足时提示调用方修正设置。
    background_score_chunks = []  # 收集真实背景像素分数，用于计算每个阈值下的 FPR。
    region_score_chunks = []  # 每个元素保存一个真实缺陷连通区域内的所有预测分数。
    minimum_score = np.inf  # 记录所有测试热图中的全局最小分数，作为阈值扫描下界。
    maximum_score = -np.inf  # 记录所有测试热图中的全局最大分数，作为阈值扫描上界。
    for _, row in test_predictions.iterrows():  # 遍历每张测试图，配对其热图和真实掩码。
        score_map = np.load(row["map_path"]).astype(np.float32)  # 载入 H×W 预测异常分数图。
        raw_mask_path = row["mask_path"]  # 先取 CSV 中保存的掩码路径或空值。
        mask_path = None if pd.isna(raw_mask_path) else raw_mask_path  # 将 CSV 缺失值还原为 None。
        true_mask = load_mask(mask_path)  # 载入二值真实掩码；正常图会得到全零掩码。
        if score_map.shape != true_mask.shape:  # 两张图每个坐标必须一一对应。
            raise ValueError(f"热图与掩码尺寸不一致：{score_map.shape} vs {true_mask.shape}")  # 尺寸错位时停止。
        minimum_score = min(minimum_score, float(score_map.min()))  # 更新所有热图的最小分数。
        maximum_score = max(maximum_score, float(score_map.max()))  # 更新所有热图的最大分数。
        background_score_chunks.append(score_map[true_mask == 0])  # 收集真实正常背景位置的分数，用来估计误报。
        component_labels = connected_components(true_mask.astype(bool), connectivity=2)  # 8 邻域连通分析，给每个缺陷区域编号。
        for component_id in range(1, int(component_labels.max()) + 1):  # 标签 0 是背景，从第一个真实缺陷区域开始。
            region_mask = component_labels == component_id  # 取出当前单独的连通缺陷区域。
            region_score_chunks.append(score_map[region_mask])  # 保存区域内分数，之后计算这个区域被覆盖的比例。
    if not region_score_chunks:  # 所有测试掩码都没有缺陷区域时，PRO 没有定义。
        raise ValueError("测试掩码中没有缺陷连通区域，无法计算 AUPRO")  # 明确提示数据或掩码问题。
    background_scores = np.sort(np.concatenate(background_score_chunks))  # 合并全部背景分数，FPR 以这些像素为分母。
    region_score_chunks = [np.sort(values) for values in region_score_chunks]
    thresholds = np.linspace(maximum_score, minimum_score, num_thresholds)  # 从高到低扫描阈值，逐步放宽异常判定。
    thresholds = np.concatenate(([np.inf], thresholds))  # 加入全无预测点，确保曲线从 FPR=0、PRO=0 起步。
    fpr_points = []  # 保存每个阈值下的背景误报率，作为曲线横坐标。
    pro_points = []  # 保存每个阈值下的区域平均覆盖率，作为曲线纵坐标。
    for threshold in thresholds:  # 每个阈值生成曲线上的一个候选点。
        fpr = float((background_scores.size - np.searchsorted(background_scores, threshold, side="left")) / background_scores.size)  # 背景中分数达到阈值的比例就是像素级 FPR。
        region_overlaps = [  # 逐个缺陷区域计算覆盖率；区域大小不会改变它在平均值中的权重。
            float((region_scores.size - np.searchsorted(region_scores, threshold, side="left")) / region_scores.size) for region_scores in region_score_chunks
        ]
        pro = float(np.mean(region_overlaps))  # 对全部缺陷区域覆盖率求平均，得到 Per-Region Overlap。
        fpr_points.append(fpr)  # 记录当前阈值的横坐标。
        pro_points.append(pro)  # 记录当前阈值的纵坐标。
    fpr_points = np.asarray(fpr_points, dtype=np.float64)  # 转数组，方便筛选与曲线面积计算。
    pro_points = np.asarray(pro_points, dtype=np.float64)  # 转数组，与 FPR 点逐一对应。
    unique_fpr = np.unique(fpr_points)  # 多个阈值可能产生相同 FPR，面积计算前整理横坐标。
    best_pro_at_fpr = np.asarray([  # 同一 FPR 可能有多个 PRO，保留其中最高覆盖率作为曲线包络。
        pro_points[fpr_points == fpr].max() for fpr in unique_fpr
    ])
    inside_limit = unique_fpr < max_fpr  # 只保留积分上限左侧的曲线点。
    curve_fpr = np.concatenate((unique_fpr[inside_limit], [max_fpr]))  # 加入准确的 FPR 积分端点。
    pro_at_limit = float(np.interp(max_fpr, unique_fpr, best_pro_at_fpr))  # 在线性曲线上插值得到端点 PRO。
    curve_pro = np.concatenate((best_pro_at_fpr[inside_limit], [pro_at_limit]))  # 组成积分范围内完整 PRO 曲线。
    aupro = float(auc(curve_fpr, curve_pro) / max_fpr)  # 积分后除以 FPR 上限，将面积归一化到 [0,1]。
    return {  # 返回指标与协议参数，便于报告中明确 AUPRO 的计算设置。
        "test_aupro": aupro,  # 归一化后的区域级定位指标。
        "aupro_max_fpr": float(max_fpr),  # 保存曲线积分的 FPR 上限，当前为 0.30。
        "aupro_num_thresholds": int(num_thresholds),  # 保存阈值采样点数，方便复现实验。
        "aupro_regions": int(len(region_score_chunks)),  # 保存参与平均的真实缺陷连通区域数。
    }


def compute_aupro(test_rows, anomaly_maps, max_fpr=0.3, bins=200):
    """兼容旧调用接口，数值定义统一到200阈值、8邻域。"""
    table = test_rows.copy()
    # evaluate_aupro读取原始预测路径，避免重存或更改已有热图。
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
    # 每类/方法最多展示分数最偏离阈值的 6 个错误，完整错误列表仍写入 CSV。
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
    print(f"指标：{results_path}")
    print(f"错误案例：{errors_path}")


if __name__ == "__main__":
    main()

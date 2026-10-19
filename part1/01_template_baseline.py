"""模板配准与像素差异图基线。"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
from PIL import Image

from common import (
    IMAGE_SIZE,
    REQUIRED_CATEGORIES,
    ensure_output_dir,
    image_score,
    load_rgb_image,
    load_visa_split,
    save_prediction_set,
    set_seed,
    split_normal_train,
)


def register_to_template(image: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, bool]:
    """用 ECC 欧氏变换做轻量配准；优化失败时返回原图并标记失败。"""
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        _, warp = cv2.findTransformECC(
            template_gray,
            image_gray,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
        height, width = template.shape[:2]
        aligned = cv2.warpAffine(
            image,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, True
    except cv2.error:
        return image, False


def build_template(training_rows, max_references: int = 24) -> tuple[np.ndarray, int]:
    """将最多 24 张正常训练图对齐到第一张参考图后求均值模板。"""
    rows = training_rows.head(max_references)
    if rows.empty:
        raise ValueError("训练划分没有正常图片，无法构造模板。")
    reference = load_rgb_image(rows.iloc[0]["image_path"], IMAGE_SIZE)
    registered = [reference.astype(np.float32)]
    for row in rows.iloc[1:].itertuples():
        image = load_rgb_image(row.image_path, IMAGE_SIZE)
        aligned, ok = register_to_template(image, reference)
        if ok:
            registered.append(aligned.astype(np.float32))
    template = np.mean(registered, axis=0).clip(0, 255).astype(np.uint8)
    return template, len(registered)


def make_difference_map(image: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, bool]:
    """配准后计算平滑的 RGB 平均绝对差异图，返回 [0,1] 浮点热图。"""
    aligned, registered = register_to_template(image, template)
    difference = np.abs(aligned.astype(np.float32) - template.astype(np.float32)).mean(axis=2) / 255.0
    anomaly_map = cv2.GaussianBlur(difference, (5, 5), 0.8)
    return anomaly_map.astype(np.float32), registered


def predict_rows(rows, template: np.ndarray) -> tuple[list[np.ndarray], list[float], int, float]:
    maps, scores = [], []
    registration_failures = 0
    started = time.perf_counter()
    for row in rows.itertuples():
        image = load_rgb_image(row.image_path, IMAGE_SIZE)
        anomaly_map, registered = make_difference_map(image, template)
        registration_failures += int(not registered)
        maps.append(anomaly_map)
        scores.append(image_score(anomaly_map))
    elapsed = time.perf_counter() - started
    return maps, scores, registration_failures, elapsed


def run_category(category: str, seed: int = 42) -> None:
    set_seed(seed)
    train_rows = load_visa_split(category, "train")
    fit_rows, validation_rows = split_normal_train(train_rows, seed=seed)
    test_rows = load_visa_split(category, "test")
    template, references_used = build_template(fit_rows)

    output_dir = ensure_output_dir("template", category)
    Image.fromarray(template).save(output_dir / "normal_template.png")
    val_maps, val_scores, val_failures, val_seconds = predict_rows(validation_rows, template)
    test_maps, test_scores, test_failures, test_seconds = predict_rows(test_rows, template)
    save_prediction_set(validation_rows, val_maps, val_scores, output_dir / "validation")
    save_prediction_set(test_rows, test_maps, test_scores, output_dir / "test")

    metadata = {
        "method": "template_ecc_absdiff",
        "category": category,
        "seed": seed,
        "image_size": list(IMAGE_SIZE),
        "reference_images_used": references_used,
        "reference_strategy": "first 24 shuffled normal training images aligned to the first image, then mean",
        "image_score": "mean of top 1% anomaly-map pixels",
        "validation_registration_failures": val_failures,
        "test_registration_failures": test_failures,
        "validation_scoring_seconds": val_seconds,
        "test_scoring_seconds": test_seconds,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[{category}] 模板基线完成；参考图 {references_used} 张；测试配准失败 {test_failures}/{len(test_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=REQUIRED_CATEGORIES, default=REQUIRED_CATEGORIES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for category in args.categories:
        run_category(category, args.seed)


if __name__ == "__main__":
    main()

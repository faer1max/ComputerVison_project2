'Normal mean-template baseline with ECC alignment and maps restored to test coordinates.'
from __future__ import annotations
import argparse
import json
import time
import cv2
import numpy as np
from PIL import Image
from common import (IMAGE_SIZE, PROTOCOL, REQUIRED_CATEGORIES, ensure_output_dir, image_score,
                    load_rgb_image, load_visa_split, save_prediction_set, set_seed,
                    split_normal_train, source_revision)


def register_to_template(image, template):
    'Return aligned RGB, the template-to-test affine matrix, and success status.'
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(template_gray, image_gray, warp, cv2.MOTION_EUCLIDEAN,
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 5)
        h, w = template.shape[:2]
        aligned = cv2.warpAffine(image, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        return aligned, warp, True
    except cv2.error:
        return image, np.eye(2, 3, dtype=np.float32), False


def build_template(training_rows, max_references=24):
    'Average up to 24 successfully aligned normal fit references.'
    rows = training_rows.head(max_references)
    if rows.empty:
        raise ValueError("Normal fit pool is empty")
    reference = load_rgb_image(rows.iloc[0].image_path, IMAGE_SIZE)
    images = [reference.astype(np.float32)]
    for row in rows.iloc[1:].itertuples():
        aligned, _, ok = register_to_template(load_rgb_image(row.image_path, IMAGE_SIZE), reference)
        if ok:
            images.append(aligned.astype(np.float32))
    return np.mean(images, axis=0).clip(0, 255).astype(np.uint8), len(images)


def restore_map(aligned_map, warp, fallback):
    'Restore a template-coordinate map; use unaligned differences at unmapped borders.'
    h, w = aligned_map.shape
    ones = np.ones((h, w), dtype=np.uint8)

    valid_template = cv2.warpAffine(ones, warp, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    valid_test = cv2.warpAffine(valid_template, warp, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    returned = cv2.warpAffine(aligned_map, warp, (w, h), flags=cv2.INTER_LINEAR)
    returned[~valid_test] = fallback[~valid_test]
    return returned.astype(np.float32), float(valid_test.mean())


def make_difference_map(image, template):
    'Return a test-coordinate float map, registration status and valid coverage.'
    aligned, warp, ok = register_to_template(image, template)
    difference = np.abs(aligned.astype(np.float32) - template.astype(np.float32)).mean(axis=2) / 255
    smoothed = cv2.GaussianBlur(difference, (5, 5), 0.8)
    raw = np.abs(image.astype(np.float32) - template.astype(np.float32)).mean(axis=2) / 255
    fallback = cv2.GaussianBlur(raw, (5, 5), 0.8)
    returned, coverage = restore_map(smoothed, warp, fallback)
    return returned, ok, coverage


def predict_rows(rows, template):
    'Predict every input row, retaining failed-registration fallbacks and timing.'
    maps, scores, quality = [], [], []
    started = time.perf_counter()
    for row in rows.itertuples():
        image = load_rgb_image(row.image_path, IMAGE_SIZE)
        anomaly_map, ok, coverage = make_difference_map(image, template)
        maps.append(anomaly_map)
        scores.append(image_score(anomaly_map))
        quality.append({"registration_ok": ok, "mapped_pixel_fraction": coverage})
    return maps, scores, quality, time.perf_counter() - started


def run_category(category, seed=42):
    'Run a single category with normal-only fitting and complete validation/test outputs.'
    set_seed(seed)
    fit, validation = split_normal_train(load_visa_split(category, "train"), seed=seed)
    test = load_visa_split(category, "test")
    started = time.perf_counter()
    template, count = build_template(fit, PROTOCOL["template_max_references"])
    construction_seconds = time.perf_counter() - started
    output = ensure_output_dir("template", category)
    Image.fromarray(template).save(output / "normal_template.png")
    timings, failures = {}, {}
    for split, rows in [("validation", validation), ("test", test)]:
        maps, scores, quality, seconds = predict_rows(rows, template)
        enriched = rows.copy()
        for column in quality[0]:
            enriched[column] = [item[column] for item in quality]
        save_prediction_set(enriched, maps, scores, output / split)
        timings[split] = seconds
        failures[split] = sum(not item["registration_ok"] for item in quality)
    metadata = {"method": "template_ecc_absdiff", "category": category, "seed": seed,
                "image_size": list(IMAGE_SIZE), "reference_images_used": count,
                "reference_images_requested": fit.head(PROTOCOL["template_max_references"]).image.tolist(),
                "template_construction_seconds": construction_seconds, "scoring_seconds": timings,
                "registration_failures": failures, "heatmap_coordinates": "original resized test image",
                "border_rule": "unaligned difference fallback for unmapped pixels; all pixels evaluated",
                "template_bytes": int(template.nbytes), "gpu_peak_memory_bytes": 0,
                "source_sha256": source_revision()}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{category}: references={count}, failures={failures}, seconds={timings}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=REQUIRED_CATEGORIES, default=REQUIRED_CATEGORIES)
    parser.add_argument("--seed", type=int, default=PROTOCOL["split_seed"])
    args = parser.parse_args()
    cv2.setNumThreads(4)
    for category in args.categories:
        run_category(category, args.seed)

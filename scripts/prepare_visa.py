"""Prepare the official VisA 1-class split and AnomalyCLIP metadata.

Uses the unmodified official SPot-the-Difference preprocessing script. Run after
prepare_assets.py has downloaded and extracted VisA; requires numpy and Pillow.
"""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/VisA"
PREPARED = ROOT / "data/VisA_pytorch"
SPLIT = ROOT / "external/spot-diff/split_csv/1cls.csv"


def main():
    with SPLIT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 10821:
        raise ValueError(f"Unexpected official CSV count: {len(rows)}")
    paths = [r["image"] for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image in official split")
    if any(r["label"] != "normal" for r in rows if r["split"] == "train"):
        raise ValueError("Training split contains anomalous images")
    for r in rows:
        with Image.open(RAW / r["image"]) as image:
            image.verify()
        if r["mask"]:
            with Image.open(RAW / r["mask"]) as mask:
                mask.verify()
    print("All 10,821 raw images and annotated masks passed image verification", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "external/spot-diff/utils/prepare_data.py"),
        "--split-type", "1cls", "--data-folder", str(RAW),
        "--save-folder", str(PREPARED), "--split-file", str(SPLIT),
    ], check=True)
    (RAW / "split_csv").mkdir(exist_ok=True)
    shutil.copyfile(SPLIT, RAW / "split_csv/1cls.csv")
    classes = sorted({r["object"] for r in rows})
    metadata = {phase: {c: [] for c in classes} for phase in ["train", "test"]}
    counts = Counter()
    for r in rows:
        category, phase = r["object"], r["split"]
        abnormal = r["label"] == "anomaly"
        label = "bad" if abnormal else "good"
        prepared_image = PREPARED / "1cls" / category / phase / label / Path(r["image"]).name
        if not prepared_image.is_file():
            raise FileNotFoundError(prepared_image)
        if abnormal:
            mask_file = PREPARED / "1cls" / category / "ground_truth/bad" / Path(r["mask"]).name
            with Image.open(mask_file) as mask:
                if not set(np.unique(np.asarray(mask))).issubset({0, 255}):
                    raise ValueError(f"Nonbinary prepared mask: {mask_file}")
        metadata[phase][category].append(dict(
            img_path=r["image"], mask_path=r["mask"] if abnormal else "",
            cls_name=category, specie_name="", anomaly=int(abnormal),
        ))
        counts[(category, phase, label)] += 1
    (RAW / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = dict(
        categories=classes, images=len(rows),
        normal=sum(r["label"] == "normal" for r in rows),
        anomalous=sum(r["label"] == "anomaly" for r in rows),
        split_counts={"/".join(k): v for k, v in sorted(counts.items())},
        raw_image_verification="passed", prepared_masks_binary=True,
        training_contains_only_normal=True,
        note="No model inference or metric evaluation has been performed.",
    )
    (ROOT / "assets/visa_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

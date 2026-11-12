'Shared VisA loading, course448 protocol, scoring and output utilities.'

from __future__ import annotations

import random
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image



PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PROJECT_ROOT / "experiment_protocol.json").read_text(encoding="utf-8"))
DATA_ROOT = Path(os.environ.get("CVPROJECT_DATA_ROOT", str(PROJECT_ROOT / "data" / "VisA"))).resolve()
SPLIT_CSV = DATA_ROOT / "split_csv" / "1cls.csv"
OUTPUT_DIR = Path(os.environ.get("CVPROJECT_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs" / "course448" / "part1"))).resolve()
REQUIRED_CATEGORIES = tuple(PROTOCOL["categories"])
IMAGE_SIZE = tuple(PROTOCOL["image_size"])


def source_revision() -> dict:
    'Hash the protocol and all Part 1 Python source files used by a run.'
    paths = [PROJECT_ROOT / "experiment_protocol.json", *sorted((PROJECT_ROOT / "part1").glob("*.py"))]
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}



def set_seed(seed: int = PROTOCOL["split_seed"]) -> None:
    'Seed Python, NumPy and PyTorch CPU/GPU random generators.'
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



def load_visa_split(
    category: str,
    split: str,
    data_root: Path | str = DATA_ROOT,
    split_csv: Path | str | None = None,
) -> pd.DataFrame:
    'Load an official split with resolved image/mask paths and binary labels.'
    if category not in REQUIRED_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {REQUIRED_CATEGORIES}")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test.")

    root = Path(data_root).expanduser().resolve()
    csv_path = Path(split_csv).expanduser().resolve() if split_csv else root / "split_csv" / "1cls.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"VisA split CSV not found: {csv_path}")

    table = pd.read_csv(csv_path)
    required_columns = {"object", "split", "label", "image", "mask"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    condition = (table["object"] == category) & (table["split"] == split)
    selected = table.loc[condition, ["object", "split", "label", "image", "mask"]].copy()
    if selected.empty:
        raise ValueError(f"Empty category {category!r} split {split!r}.")
    if not selected["label"].isin(["normal", "anomaly"]).all():
        raise ValueError("CSV labels must be normal or anomaly.")
    if split == "train" and not selected["label"].eq("normal").all():
        raise ValueError("The official train split must contain only normal images.")

    def resolve_relative_path(value: object) -> Path | None:
        if pd.isna(value) or not str(value).strip():
            return None
        relative = Path(*str(value).replace("\\", "/").split("/"))
        return root / relative

    selected["image_path"] = selected["image"].map(resolve_relative_path)
    selected["mask_path"] = selected["mask"].map(resolve_relative_path)
    selected["label_id"] = selected["label"].map({"normal": 0, "anomaly": 1}).astype("int64")

    missing_images = [p for p in selected["image_path"] if p is None or not p.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Invalid image paths: {len(missing_images)}; example: {missing_images[0]}")
    anomaly_rows = selected["label"].eq("anomaly")
    missing_masks = [p for p in selected.loc[anomaly_rows, "mask_path"] if p is None or not p.is_file()]
    if missing_masks:
        raise FileNotFoundError(f"Invalid anomaly mask paths: {len(missing_masks)}; example: {missing_masks[0]}")

    return selected.reset_index(drop=True)


def split_normal_train(
    train_rows: pd.DataFrame,
    validation_fraction: float = PROTOCOL["validation_fraction"],
    seed: int = PROTOCOL["split_seed"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    'Split nonempty normal training rows into disjoint fit and validation sets.'
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_rows.empty or not train_rows["label"].eq("normal").all():
        raise ValueError("Validation splitting requires nonempty normal-only rows.")

    n_validation = max(1, round(len(train_rows) * validation_fraction))
    if n_validation >= len(train_rows):
        raise ValueError("Not enough normal rows for both fit and validation subsets.")
    order = np.random.default_rng(seed).permutation(len(train_rows))
    validation = train_rows.iloc[order[:n_validation]].reset_index(drop=True)
    training = train_rows.iloc[order[n_validation:]].reset_index(drop=True)
    return training, validation



def load_rgb_image(
    image_path: Path | str,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    'Read RGB and return a resized H/W/C uint8 array.'
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_mask(
    mask_path: Path | str | None,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    'Read a nearest-neighbor binary mask; normal images receive zeros.'
    width, height = image_size
    if mask_path is None or pd.isna(mask_path):
        return np.zeros((height, width), dtype=np.uint8)
    with Image.open(mask_path) as mask:
        mask = mask.convert("L").resize(image_size, Image.Resampling.NEAREST)

        return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def image_to_tensor(image: np.ndarray):
    'Convert RGB uint8 H/W/C to floating C/H/W in the range [0, 1].'
    import torch

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected H/W/3 RGB input; received {image.shape}")

    writable_image = np.array(image, copy=True, order="C")
    return torch.from_numpy(writable_image).permute(2, 0, 1).float().div_(255.0)


def ensure_output_dir(*parts: str) -> Path:
    'Create a directory beneath the configured Part 1 output root.'
    path = OUTPUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_score(anomaly_map: np.ndarray, top_fraction: float = 0.01) -> float:
    'Mean of the highest ceil(1 percent) map values.'
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    values = np.asarray(anomaly_map, dtype=np.float32).ravel()
    count = max(1, int(np.ceil(values.size * top_fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def save_prediction_set(
    rows: pd.DataFrame,
    anomaly_maps: list[np.ndarray],
    scores: list[float],
    output_dir: Path | str,
) -> Path:
    'Save float32 maps and their prediction-index CSV.'
    if len(rows) != len(anomaly_maps) or len(rows) != len(scores):
        raise ValueError("rows, maps and scores must have equal lengths.")
    output_dir = Path(output_dir)
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    result = rows.copy().reset_index(drop=True)
    map_paths = []
    for index, (row, anomaly_map) in enumerate(zip(result.itertuples(), anomaly_maps)):
        filename = f"{index:04d}_{Path(row.image_path).stem}.npy"
        map_path = map_dir / filename
        np.save(map_path, np.asarray(anomaly_map, dtype=np.float32))
        map_paths.append(map_path)
    result["score"] = np.asarray(scores, dtype=np.float32)
    result["map_path"] = map_paths
    csv_path = output_dir / "predictions.csv"
    result.to_csv(csv_path, index=False)
    return csv_path

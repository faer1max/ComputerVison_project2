"""Part 1 共用设置、VisA 数据读取与结果保存工具。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# %% 路径与数据配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "VisA"
SPLIT_CSV = DATA_ROOT / "split_csv" / "1cls.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part1"
REQUIRED_CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"
)
IMAGE_SIZE = (256, 256)  # Pillow 参数顺序为 (宽, 高)


# %% 可复现设置
def set_seed(seed: int = 42) -> None:
    """固定 Python、NumPy、PyTorch CPU/GPU 的随机数序列。"""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# %% 官方 VisA 1-class CSV
def load_visa_split(
    category: str,
    split: str,
    data_root: Path | str = DATA_ROOT,
    split_csv: Path | str | None = None,
) -> pd.DataFrame:
    """读一个类别的官方 train/test 划分并补充可直接访问的路径列。

    返回列包含原始 object/split/label/image/mask，以及 image_path、mask_path、label_id。
    正常图片没有 mask_path；异常图片的 mask_path 指向像素级标注文件。
    """
    if category not in REQUIRED_CATEGORIES:
        raise ValueError(f"未知类别 {category!r}，可选值为 {REQUIRED_CATEGORIES}")
    if split not in {"train", "test"}:
        raise ValueError("split 只能是 'train' 或 'test'。")

    root = Path(data_root).expanduser().resolve()
    csv_path = Path(split_csv).expanduser().resolve() if split_csv else root / "split_csv" / "1cls.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 VisA 划分 CSV：{csv_path}")

    table = pd.read_csv(csv_path)
    required_columns = {"object", "split", "label", "image", "mask"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"CSV 缺少字段：{sorted(missing_columns)}")

    condition = (table["object"] == category) & (table["split"] == split)
    selected = table.loc[condition, ["object", "split", "label", "image", "mask"]].copy()
    if selected.empty:
        raise ValueError(f"类别 {category!r} 的 {split!r} 数据为空。")
    if not selected["label"].isin(["normal", "anomaly"]).all():
        raise ValueError("CSV 的 label 列包含 normal/anomaly 以外的值。")
    if split == "train" and not selected["label"].eq("normal").all():
        raise ValueError("官方 train 划分应只包含正常图，请检查 CSV。")

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
        raise FileNotFoundError(f"有 {len(missing_images)} 张图片路径无效，例如：{missing_images[0]}")
    anomaly_rows = selected["label"].eq("anomaly")
    missing_masks = [p for p in selected.loc[anomaly_rows, "mask_path"] if p is None or not p.is_file()]
    if missing_masks:
        raise FileNotFoundError(f"有 {len(missing_masks)} 张异常图缺少有效掩码，例如：{missing_masks[0]}")

    return selected.reset_index(drop=True)


def split_normal_train(
    train_rows: pd.DataFrame,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只将正常官方训练图划分为训练子集和验证子集。"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间。")
    if train_rows.empty or not train_rows["label"].eq("normal").all():
        raise ValueError("只能从非空且全为正常图的数据中划分验证集。")

    n_validation = max(1, round(len(train_rows) * validation_fraction))
    if n_validation >= len(train_rows):
        raise ValueError("样本数太少，无法同时保留训练图和验证图。")
    order = np.random.default_rng(seed).permutation(len(train_rows))
    validation = train_rows.iloc[order[:n_validation]].reset_index(drop=True)
    training = train_rows.iloc[order[n_validation:]].reset_index(drop=True)
    return training, validation


# %% 图像与掩码预处理
def load_rgb_image(
    image_path: Path | str,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """读 RGB 图并缩放，返回 uint8、H×W×3 数组。"""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_mask(
    mask_path: Path | str | None,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """读二值掩码并缩放；无掩码的正常图返回全零数组。"""
    width, height = image_size
    if mask_path is None or pd.isna(mask_path):
        return np.zeros((height, width), dtype=np.uint8)
    with Image.open(mask_path) as mask:
        mask = mask.convert("L").resize(image_size, Image.Resampling.NEAREST)
        # VisA 部分掩码像素值为 1 或 6；任何正值都代表缺陷，不能用 >127。
        return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def image_to_tensor(image: np.ndarray):
    """将 RGB uint8 数组转成 PyTorch 的 C×H×W 浮点张量，范围 [0, 1]。"""
    import torch

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"需要 H×W×3 RGB 图片，收到形状 {image.shape}")
    # np.asarray(PIL_image) 可能是只读视图；显式复制，避免 torch.from_numpy 警告。
    writable_image = np.array(image, copy=True, order="C")
    return torch.from_numpy(writable_image).permute(2, 0, 1).float().div_(255.0)


def ensure_output_dir(*parts: str) -> Path:
    """在 outputs/part1 下创建并返回指定结果子目录。"""
    path = OUTPUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_score(anomaly_map: np.ndarray, top_fraction: float = 0.01) -> float:
    """取热图中最高的一小部分像素的均值，作为整张图的异常分数。"""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction 必须在 (0, 1] 范围内。")
    values = np.asarray(anomaly_map, dtype=np.float32).ravel()
    count = max(1, int(np.ceil(values.size * top_fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def save_prediction_set(
    rows: pd.DataFrame,
    anomaly_maps: list[np.ndarray],
    scores: list[float],
    output_dir: Path | str,
) -> Path:
    """保存每张图的异常热图和索引 CSV，供评估脚本复用。"""
    if len(rows) != len(anomaly_maps) or len(rows) != len(scores):
        raise ValueError("rows、anomaly_maps 和 scores 的数量必须一致。")
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

\n
"""课程统一输入：共享几何变换，骨干归一化显式作为模型接口。"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

SIZE = 448  # 同时是 patch=14 和 AE 下采样倍率=16 的整数倍。
RGB_SIZE = (SIZE, SIZE)  # Pillow 的顺序为宽、高。
DINO_MEAN, DINO_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def shared_rgb(image, image_size=RGB_SIZE):
    """Path/str、PIL 或 RGB ndarray → 共享的 448×448×3 uint8 RGB 数组。"""
    if tuple(image_size) != RGB_SIZE:
        raise ValueError("课程模式禁止覆盖 448×448 统一尺寸")
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            source = source.convert("RGB")
    elif isinstance(image, np.ndarray):
        source = Image.fromarray(image).convert("RGB")
    else:
        source = image.convert("RGB")
    return np.array(source.resize(RGB_SIZE, Image.Resampling.BILINEAR), copy=True)


def shared_mask(mask_path, image_size=RGB_SIZE):
    """掩码路径或 None → 448×448 uint8 0/1；缩放只使用最近邻。"""
    if tuple(image_size) != RGB_SIZE:
        raise ValueError("课程模式禁止覆盖掩码尺寸")
    if mask_path is None:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    with Image.open(mask_path) as source:
        return (np.asarray(source.convert("L").resize(RGB_SIZE, Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def model_tensor(image, model):
    """共享 RGB 图 → C/H/W Tensor；只在模型入口应用权重要求的仿射归一化。"""
    rgb = shared_rgb(image)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    if model == "dino":
        tensor = transforms.functional.normalize(tensor, DINO_MEAN, DINO_STD)
    elif model == "clip":
        tensor = transforms.functional.normalize(tensor, CLIP_MEAN, CLIP_STD)
    elif model not in ("template", "autoencoder"):
        raise ValueError(model)
    return tensor


def clip_mask_tensor(image):
    """官方 Dataset 已提供 PIL 二值掩码；按共享最近邻规则转为 1/H/W Tensor。"""
    mask = np.asarray(image.convert("L").resize(RGB_SIZE, Image.Resampling.NEAREST)) > 0
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def dino_prepare_image(self, image):
    """替换官方 prepare_image 接口，返回统一 Tensor 与 32×32 patch 网格。"""
    return model_tensor(image, "dino"), (SIZE // 14, SIZE // 14)


def clip_transform(args):
    """替换官方 get_transform 接口，阻止内部再次使用 bicubic/center crop。"""
    if args.image_size != SIZE:
        raise ValueError("课程 CLIP 输入必须是 448")
    return lambda image: model_tensor(image, "clip"), clip_mask_tensor


def configure_part1(common, output_root):
    """将原有基线/评估模块共享的 common 接口接到 448 设置及独立输出目录。"""
    common.IMAGE_SIZE = RGB_SIZE
    common.OUTPUT_DIR = Path(output_root)
    common.load_rgb_image = shared_rgb
    common.load_mask = shared_mask


PROTOCOL = {
    "schema_version": 1,
    "image_size_wh": list(RGB_SIZE),
    "geometry": "Pillow RGB -> bilinear resize to 448x448, no crop or padding",
    "mask": "Pillow L -> nearest resize to 448x448 -> greater than zero",
    "model_interface_normalization": {
        "template": "uint8 RGB", "autoencoder": "float RGB / 255",
        "dino": {"mean": DINO_MEAN, "std": DINO_STD},
        "clip": {"mean": CLIP_MEAN, "std": CLIP_STD},
    },
    "model_specific_steps": {
        "template": "ECC registration, unregistered fallback on failure",
        "dino": "official agnostic foreground masking and 8 reference rotations",
        "clip": "official DPAM, pretrained MVTec prompt, gaussian sigma=4",
    },
    "evaluation": {"image_fpr_target": 0.05, "pixel_fpr_target": 0.01,
                   "threshold_source": "normal validation only", "aupro_max_fpr": 0.3,
                   "aupro_thresholds": 200, "connectivity": 8, "std_ddof": 1},
    "status": "input_contract_only_not_full_experiment",
}

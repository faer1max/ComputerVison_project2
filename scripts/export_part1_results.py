"""导出可分享结果，不复制模型、完整热图或机器绝对路径。"""
from pathlib import Path
import json
import shutil
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
import common
from verify_part1_release import verify_release


def export_results():
    """核查通过的完整输出→公开汇总和24例视觉检查面板，返回目的目录。"""
    verification = verify_release()
    destination = ROOT / "results/part1_course448"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(common.OUTPUT_DIR / "evaluation/metrics.csv")
    metrics.to_csv(destination / "metrics.csv", index=False)
    metric_names = ["image_auroc", "image_ap", "image_fpr_at_threshold", "pixel_auroc", "pixel_ap", "aupro"]
    metrics.groupby("method")[metric_names].mean().to_csv(destination / "macro_mean.csv")
    (destination / "release_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    shutil.copy2(common.OUTPUT_DIR / "evaluation/evaluation_protocol.json", destination / "evaluation_protocol.json")
    metadata = {}
    for method in ["template", "autoencoder"]:
        metadata[method] = {}
        for category in common.REQUIRED_CATEGORIES:
            directory = common.OUTPUT_DIR / method / category
            metadata[method][category] = json.loads((directory / "run_metadata.json").read_text())
            if method == "autoencoder":
                shutil.copy2(directory / "training_history.csv", destination / f"{category}_training_history.csv")
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    errors = pd.read_csv(common.OUTPUT_DIR / "evaluation/error_cases.csv")
    errors["image"] = [Path(path).relative_to(common.DATA_ROOT).as_posix() for path in errors.image_path]
    errors.drop(columns=["image_path"]).to_csv(destination / "error_cases.csv", index=False)
    errors.groupby(["method", "category", "error_type"]).size().rename("count").to_csv(destination / "error_counts.csv")
    # 每种方法/类别优先展示一个FP和一个FN；总24个不同方法-图像案例。
    selected = []
    for method in ["template", "autoencoder"]:
        for category in common.REQUIRED_CATEGORIES:
            group = errors.query("method == @method and category == @category")
            chosen = []
            for kind in ["false_positive", "false_negative"]:
                subset = group[group.error_type == kind]
                if not subset.empty:
                    chosen.append(subset.iloc[0])
            for _, row in group.iterrows():
                if len(chosen) >= 2:
                    break
                if row.image not in [item.image for item in chosen]:
                    chosen.append(row)
            selected.extend(chosen)
    cases = pd.DataFrame(selected).reset_index(drop=True)
    cases.insert(0, "case_id", [f"C{i+1:02}" for i in range(len(cases))])
    cases.drop(columns=["image_path"]).to_csv(destination / "visual_cases.csv", index=False)
    for page in range((len(cases) + 5) // 6):
        subset = cases.iloc[page * 6:(page + 1) * 6]
        fig, axes = plt.subplots(len(subset), 4, figsize=(12, 3 * len(subset)), squeeze=False)
        for index, case in enumerate(subset.itertuples()):
            predictions = pd.read_csv(common.OUTPUT_DIR / case.method / case.category / "test/predictions.csv")
            record = predictions[predictions.image == case.image].iloc[0]
            image = common.load_rgb_image(record.image_path)
            mask = common.load_mask(None if pd.isna(record.mask_path) else record.mask_path)
            heatmap = np.load(record.map_path)
            threshold = metrics.query("method == @case.method and category == @case.category").iloc[0].pixel_threshold
            for column, (panel, cmap) in enumerate([(image, None), (mask, "gray"), (heatmap, "inferno"), (heatmap > threshold, "gray")]):
                axes[index, column].imshow(panel, cmap=cmap)
                axes[index, column].axis("off")
            axes[index, 0].set_title(f"{case.case_id} {case.method}/{case.category}\n{case.error_type}: {Path(case.image).name}", fontsize=9)
        for column, name in enumerate(["Input", "Ground truth", "Heatmap", "Thresholded"]):
            axes[0, column].set_xlabel(name)
        fig.tight_layout()
        fig.savefig(destination / f"visual_cases_{page+1}.png", dpi=120)
        plt.close(fig)
    import torch, cv2, scipy, skimage, sklearn, PIL
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
                   "scipy": scipy.__version__, "scikit_image": skimage.__version__,
                   "scikit_learn": sklearn.__version__, "pillow": PIL.__version__,
                   "matplotlib": matplotlib.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (destination / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(export_results())

'Predict a single image without reading its ground truth.'
from pathlib import Path
import argparse
import importlib.util
import json
import io
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
from common import OUTPUT_DIR, REQUIRED_CATEGORIES, image_score, load_rgb_image, image_to_tensor


def predict_image(method, category, image_path, destination):
    'Return label-free prediction metadata and save image/heatmap panels.'
    image = load_rgb_image(image_path)
    if method == "template":
        spec = importlib.util.spec_from_file_location("demo_template", ROOT / "part1/01_template_baseline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = np.array(Image.open(OUTPUT_DIR / method / category / "normal_template.png").convert("RGB"))
        anomaly_map, _, _ = module.make_difference_map(image, reference)
    else:
        import torch
        spec = importlib.util.spec_from_file_location("demo_ae", ROOT / "part1/02_autoencoder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ConvAutoencoder().eval()
        checkpoint = torch.load(OUTPUT_DIR / method / category / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        tensor = image_to_tensor(image).unsqueeze(0)
        with torch.inference_mode():
            anomaly_map = (tensor - model(tensor)).abs().mean(dim=1)[0].numpy()
    metrics = pd.read_csv(OUTPUT_DIR / "evaluation/metrics.csv")
    calibration = metrics.query("category == @category and method == @method").iloc[0]
    raw_score = image_score(anomaly_map)

    score = float(pd.read_csv(io.StringIO("score\n" + str(np.float32(raw_score)))).score.iloc[0])
    threshold = float(calibration.image_threshold)
    result = {"method": method, "category": category, "score": score,
              "raw_float32_score": raw_score, "score_representation": "original float32 CSV decimal",
              "threshold": threshold, "decision": "NG" if score > threshold else "OK",
              "threshold_source": "normal validation only"}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "heatmap.npy", anomaly_map)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(image)
    axes[1].imshow(image)
    axes[1].imshow(anomaly_map, cmap="inferno", alpha=0.65)
    axes[2].imshow(anomaly_map > float(calibration.pixel_threshold), cmap="gray")
    for ax, title in zip(axes, ["Input", "Anomaly heatmap", "Thresholded mask"]):
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{category}/{method}: {result['decision']}, score={score:.4f}, threshold={threshold:.4f}")
    fig.tight_layout()
    fig.savefig(destination / "prediction.png", dpi=140)
    plt.close(fig)
    (destination / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["template", "autoencoder"], required=True)
    parser.add_argument("--category", choices=REQUIRED_CATEGORIES, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "demo")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.method, args.category, args.image, args.output), indent=2))

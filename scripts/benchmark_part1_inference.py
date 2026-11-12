'Measure both baselines sequentially on the same normal validation images.'
from pathlib import Path
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
import common


def import_method(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "part1" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    template = import_method("benchmark_template", "01_template_baseline.py")
    ae = import_method("benchmark_ae", "02_autoencoder.py")
    rows, inputs = [], {}
    for category in common.REQUIRED_CATEGORIES:
        _, validation = common.split_normal_train(common.load_visa_split(category, "train"))
        samples = validation.head(16)
        inputs[category] = samples.image.tolist()
        for method in ["template", "autoencoder"]:
            directory = common.OUTPUT_DIR / method / category
            if method == "template":
                reference = common.load_rgb_image(directory / "normal_template.png")
                model = None
            else:
                model = ae.ConvAutoencoder().cuda().eval()
                model.load_state_dict(torch.load(directory / "model.pt", map_location="cuda", weights_only=True)["model_state_dict"])
                with torch.inference_mode():
                    for path in samples.image_path.iloc[:3]:
                        model(common.image_to_tensor(common.load_rgb_image(path)).unsqueeze(0).cuda())
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            preparation, features, scoring, total = [], [], [], []
            with torch.inference_mode():
                for path in samples.image_path:
                    start = time.perf_counter()
                    image = common.load_rgb_image(path)
                    if model is None:
                        preparation.append(time.perf_counter() - start)
                        stage = time.perf_counter()
                        heatmap, _, _ = template.make_difference_map(image, reference)
                        common.image_score(heatmap)
                        scoring.append(time.perf_counter() - stage)
                    else:
                        tensor = common.image_to_tensor(image).unsqueeze(0).cuda()
                        torch.cuda.synchronize()
                        preparation.append(time.perf_counter() - start)
                        stage = time.perf_counter()
                        encoded = model.encoder(tensor)
                        torch.cuda.synchronize()
                        features.append(time.perf_counter() - stage)
                        stage = time.perf_counter()
                        reconstructed = model.decoder(encoded)
                        heatmap = (tensor - reconstructed).abs().mean(1)[0].cpu().numpy()
                        common.image_score(heatmap)
                        scoring.append(time.perf_counter() - stage)
                        del tensor, encoded, reconstructed
                    total.append(time.perf_counter() - start)
            rows.append({"method": method, "category": category, "normal_validation_images": len(samples),
                         "preprocessing_mean_ms": np.mean(preparation) * 1000,
                         "encoder_mean_ms": np.mean(features) * 1000 if features else None,
                         "scoring_mean_ms": np.mean(scoring) * 1000,
                         "total_mean_ms": np.mean(total) * 1000,
                         "inference_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if model is not None else 0,
                         "template_or_parameter_bytes": int(reference.nbytes) if model is None else sum(p.numel() * p.element_size() for p in model.parameters())})
            if model is not None:
                del model
                torch.cuda.empty_cache()
    destination = ROOT / "results/part1_course448"
    pd.DataFrame(rows).to_csv(destination / "inference_efficiency.csv", index=False)
    (destination / "efficiency_protocol.json").write_text(json.dumps({
        "samples": inputs, "warmup": "AE first three normal validation images; excluded",
        "ae_device": torch.cuda.get_device_name(0), "cpu_threads": 4,
        "source_sha256": common.source_revision(),
        "scope": "16 fixed normal validation images per category; sequential methods; no speed ranking claim",
        "template_encoder": "not applicable; ECC/difference included in scoring",
        "gpu_memory": "PyTorch allocated inference peak, includes parameters; separate from training peak"
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))

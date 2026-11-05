"""上传前核对全量输出、源码身份、数据隔离和图像指标；不生成训练结果。"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
import common


def verify_release():
    """返回机器可读核查报告；任何源码、样本、热图或指标不一致立即停止。"""
    metrics = pd.read_csv(common.OUTPUT_DIR / "evaluation/metrics.csv")
    expected_hashes = common.source_revision()
    checks = []
    for method in ["template", "autoencoder"]:
        for category in common.REQUIRED_CATEGORIES:
            directory = common.OUTPUT_DIR / method / category
            metadata = json.loads((directory / "run_metadata.json").read_text())
            if metadata["source_sha256"] != expected_hashes:
                raise ValueError(f"源码身份不一致: {method}/{category}")
            if metadata["image_size"] != list(common.IMAGE_SIZE):
                raise ValueError("输入尺寸不一致")
            if method == "autoencoder" and metadata["epochs_requested"] != common.PROTOCOL["autoencoder_epochs"]:
                raise ValueError("AE训练轮次与协议不同")
            if method == "autoencoder":
                import torch
                history = pd.read_csv(directory / "training_history.csv")
                expected_epochs = common.PROTOCOL["autoencoder_epochs"]
                if history.epoch.tolist() != list(range(1, expected_epochs + 1)):
                    raise ValueError("训练历史缺轮次")
                best_epoch = int(history.loc[history.validation_mse.idxmin(), "epoch"])
                checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
                if (best_epoch != metadata["best_epoch"] or checkpoint["best_epoch"] != best_epoch
                        or checkpoint["image_size"] != list(common.IMAGE_SIZE)
                        or checkpoint["category"] != category):
                    raise ValueError("最佳轮次/检查点/输入协议不一致")
            fit, validation = common.split_normal_train(common.load_visa_split(category, "train"))
            test = common.load_visa_split(category, "test")
            if set(fit.image) & set(validation.image) or set(fit.image) & set(test.image):
                raise ValueError("数据泄漏")
            count = 0
            tables = {}
            for split, expected in [("validation", validation), ("test", test)]:
                predictions = pd.read_csv(directory / split / "predictions.csv")
                if set(predictions.image) != set(expected.image) or len(predictions) != len(expected):
                    raise ValueError(f"输出样本不完整: {method}/{category}/{split}")
                for row in predictions.itertuples():
                    anomaly_map = np.load(row.map_path)
                    if anomaly_map.shape != common.IMAGE_SIZE[::-1] or not np.isfinite(anomaly_map).all():
                        raise ValueError(f"异常热图: {row.map_path}")
                    if not np.isclose(common.image_score(anomaly_map), row.score, atol=1e-6, rtol=1e-5):
                        raise ValueError("CSV分数与热图不同")
                    count += 1
                tables[split] = predictions
            row = metrics.query("method == @method and category == @category").iloc[0]
            validation, test = tables["validation"], tables["test"]
            threshold = float(np.quantile(validation.score, 0.95, method="higher"))
            labels = test.label.eq("anomaly").to_numpy()
            expected = {"image_auroc": roc_auc_score(labels, test.score),
                        "image_ap": average_precision_score(labels, test.score),
                        "image_fpr_at_threshold": float((test.loc[~labels, "score"] > threshold).mean())}
            for key, value in expected.items():
                if not np.isclose(row[key], value):
                    raise ValueError(f"图像指标复算不一致: {key}")
            for key in ["pixel_auroc", "pixel_ap", "aupro"]:
                if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                    raise ValueError(f"像素指标数值无效: {key}")
            if float((validation.score > threshold).mean()) > 0.05 + 1e-9:
                raise ValueError("验证误报率超过预定上限")
            checks.append({"method": method, "category": category, "finite_maps_verified": count})
    return {"status": "passed", "source_sha256": expected_hashes, "checks": checks,
            "total_maps_verified": sum(item["finite_maps_verified"] for item in checks)}


if __name__ == "__main__":
    result = verify_release()
    path = common.OUTPUT_DIR / "evaluation/release_verification.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

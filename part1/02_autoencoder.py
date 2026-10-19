"""只使用正常训练图的卷积自编码器基线。"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import (
    IMAGE_SIZE,
    REQUIRED_CATEGORIES,
    ensure_output_dir,
    image_score,
    image_to_tensor,
    load_rgb_image,
    load_visa_split,
    save_prediction_set,
    set_seed,
    split_normal_train,
)


class NormalImageDataset(Dataset):
    """只返回图像张量，确保训练过程不会使用测试掩码或标签。"""

    def __init__(self, rows):
        self.paths = rows["image_path"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return image_to_tensor(load_rgb_image(self.paths[index], IMAGE_SIZE))


class ConvAutoencoder(nn.Module):
    """四级卷积编码器和对称转置卷积解码器，输入输出均为 256×256 RGB。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


def predict_rows(model: nn.Module, rows, device: torch.device) -> tuple[list[np.ndarray], list[float], float]:
    model.eval()
    maps, scores = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows.itertuples():
            image = load_rgb_image(row.image_path, IMAGE_SIZE)
            tensor = image_to_tensor(image).unsqueeze(0).to(device)
            reconstruction = model(tensor)
            anomaly_map = (tensor - reconstruction).abs().mean(dim=1)[0].cpu().numpy()
            maps.append(anomaly_map.astype(np.float32))
            scores.append(image_score(anomaly_map))
    return maps, scores, time.perf_counter() - started


def run_category(
    category: str,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    official_train = load_visa_split(category, "train")
    train_rows, validation_rows = split_normal_train(official_train, seed=seed)
    test_rows = load_visa_split(category, "test")
    train_loader = DataLoader(
        NormalImageDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=(device.type == "cuda"),
    )
    validation_loader = DataLoader(
        NormalImageDataset(validation_rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for images in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images)
            loss = loss_function(reconstruction, images)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for images in validation_loader:
                images = images.to(device, non_blocking=True)
                validation_losses.append(float(loss_function(model(images), images).cpu()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{category}] epoch {epoch:02d}/{epochs}: train={train_loss:.6f}, val={validation_loss:.6f}")

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("训练没有产生有效的模型检查点。")
    model.load_state_dict(best_state)

    output_dir = ensure_output_dir("autoencoder", category)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "category": category,
            "seed": seed,
            "image_size": list(IMAGE_SIZE),
            "best_epoch": best_epoch,
            "validation_mse": best_validation_loss,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    val_maps, val_scores, val_seconds = predict_rows(model, validation_rows, device)
    test_maps, test_scores, test_seconds = predict_rows(model, test_rows, device)
    save_prediction_set(validation_rows, val_maps, val_scores, output_dir / "validation")
    save_prediction_set(test_rows, test_maps, test_scores, output_dir / "test")
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metadata = {
        "method": "normal_only_conv_autoencoder",
        "category": category,
        "seed": seed,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss": "mean squared reconstruction error",
        "image_score": "mean of top 1% absolute residual pixels",
        "device": str(device),
        "training_seconds": training_seconds,
        "validation_scoring_seconds": val_seconds,
        "test_scoring_seconds": test_seconds,
        "gpu_peak_memory_bytes": peak_memory,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[{category}] 自编码器完成；最佳 epoch={best_epoch}；设备={device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=REQUIRED_CATEGORIES, default=REQUIRED_CATEGORIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    for category in args.categories:
        run_category(category, args.seed, args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()

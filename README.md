# Few-Shot Industrial Visual Anomaly Detection

Final project for AIAA 3201 Introduction to Computer Vision.

## Goal

Given a small set of defect-free reference images, this project builds a visual anomaly detection system that:

- predicts an image-level anomaly score,
- makes an OK/NG decision,
- localizes anomalous regions at pixel level with a heatmap,
- remains robust under changes in illumination, viewpoint, background, and product category.

## Planned Scope

- Classical normality modeling baseline:
  - image registration to normal templates,
  - hand-crafted difference maps,
  - thresholding based only on normal validation data,
  - convolutional autoencoder reconstruction baseline.
- Foundation-model reproduction:
  - AnomalyDINO with DINOv2 patch features,
  - 1/2/4/8-shot memory-bank retrieval,
  - AnomalyCLIP zero-shot comparison.
- Exploration:
  - robustness and reliability improvement such as multi-scale features, background suppression, feature/prompt fusion, memory compression, or selective prediction.

## Dataset

The required dataset is VisA. The course benchmark categories are:

- candle
- capsules
- cashew
- chewinggum
- fryum
- pipe fryum

Evaluation on all 12 VisA categories is a stretch goal.

## Metrics

Image-level detection:

- AUROC
- Average Precision
- false-positive rate at the selected operating threshold

Pixel-level localization:

- pixel AUROC
- pixel AP
- AUPRO

Few-shot stability:

- mean and standard deviation over at least three random reference selections for 1/2/4/8-shot settings

Efficiency:

- feature extraction time
- scoring time
- peak GPU memory
- memory-bank size

## Project Layout

The repository root is `D:/MinghuaXiong/computervision_project` (the CV project folder).

- `part1/`: classical template/difference and reconstruction baselines.
- `part2/`: AnomalyDINO reproduction and AnomalyCLIP comparison.
- `part3/`: robustness improvements, controlled ablations, and failure analysis.
- `literature/`: the ten recommended papers grouped by topic, sources, and reading notes.
- `data/`, `weights/`, `external/`: shared datasets, model weights, and official repositories.
- `scripts/`: shared preparation utilities.
- `assets/`: download inventory, source revisions, data verification, and report template.
- `outputs/part1/`, `outputs/part2/`, `outputs/part3/`: per-part results, created when experiments run and excluded from Git.

Each part has its own README. Keep the same data split and evaluation protocol across parts. Shared assets remain at the root to avoid duplication. See `PREPARATION.md` for downloaded asset details and `PROJECT_NOTES.md` for course requirements.

## Repository Status

Part 1 now contains runnable template-difference and normal-only convolutional autoencoder baselines, with shared VisA loading and evaluation utilities. See `part1/README.md` for setup, commands, metric protocol, and output locations. Parts 2 and 3 remain planned work; their folders contain project notes only.


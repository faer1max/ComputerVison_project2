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

## Repository Status

This repository is being prepared for the final project. Code, setup instructions, data preparation steps, model checkpoints, evaluation commands, and representative visual results will be added as the implementation progresses.


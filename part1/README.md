# Part 1: course448 classical baselines

Use [PREPARATION](../PREPARATION.md) first: Python 3.11, compatible torch/torchvision, Part 1 dependencies, pinned upstream sources, VisA and its official split.

## Train, infer and evaluate

```powershell
python part1/verify.py
python part1/01_template_baseline.py
python part1/02_autoencoder.py
python part1/03_evaluate.py
python scripts/verify_part1_release.py
python scripts/export_part1_results.py
python scripts/analyze_visual_cases.py
python scripts/benchmark_part1_inference.py
```

The default output root is `outputs/course448/part1`. Use `CVPROJECT_DATA_ROOT` for an existing raw VisA directory and `CVPROJECT_OUTPUT_ROOT` for an existing or separate Part 1 output root. Keep historical 256 runs separate. The template uses up to 24 normal fit images; ECC-aligned successful references are averaged. The test map is restored to test coordinates, with unaligned differences at unmapped borders and failed registration retained. The AE trains on all normal fit images for 10 epochs (seed 42, batch 16, Adam lr 0.001); the best normal-validation MSE checkpoint is saved as `autoencoder/<category>/model.pt`.

Scores are top-1% heatmap means. Only normal validation chooses image/pixel thresholds. Shared sizes, masks and AUPRO settings are in `experiment_protocol.json`. Template alignment/smoothing and AE reconstruction are method-specific steps, not hidden common preprocessing.

## Fast checks and safe output handling

For a quick one-category experiment, set a NEW `CVPROJECT_OUTPUT_ROOT` before using `--categories candle` or AE `--epochs 1 --batch-size 8`. Such outputs are not formal ten-epoch results and must not overwrite the formal output directory. `--help` documents the CLI options. The full verifier rejects incomplete rows, mismatched source identity or training configuration.

## Single-image demo

First generate the template/AE outputs and run evaluation. The demo needs their saved templates/checkpoints and `evaluation/metrics.csv`:

```powershell
python scripts/part1_demo.py --method template --category candle --image data/VisA/candle/Data/Images/Anomaly/000.JPG --output outputs/demo_template
python scripts/part1_demo.py --method autoencoder --category candle --image data/VisA/candle/Data/Images/Anomaly/000.JPG --output outputs/demo_autoencoder
```

Outputs: `prediction.json`, `prediction.png`, `heatmap.npy`. The JSON includes score, fixed threshold and OK/NG. Pixel decisions use a separate pixel threshold, so OK may coexist with nonempty predicted pixels. The example is anomalous and the published baseline demos miss it; a running demo is not proof of correct detection.

Part 1 CSVs originally stored scores as float32 shortest decimals. The demo and supplementary brightness study retain this calibrated representation and also expose the raw score, avoiding floating-point flips for threshold ties without moving the threshold.

## Supplementary analysis

```powershell
python scripts/illumination_course448.py
python scripts/build_part12_comparison.py
python scripts/verify_part12_supplement.py
```

The brightness study uses all 324 normal validation images, multipliers 0.75/1.0/1.25 and unchanged templates, checkpoints and thresholds. It does not use test labels or measure defect recall. See [analysis](../docs/PART1_PART2_ANALYSIS.md), [requirement checklist](../docs/PART1_PART2_REQUIREMENT_CHECKLIST.md), and [unified results](../results/part12_comparison/README.md).

Full maps/checkpoints are excluded from Git; regenerate them above. The shared results directory contains metrics, histories, metadata, classified failures and representative panels. Resource times have explicitly different stage boundaries and must not be treated as an isolated end-to-end speed ranking.

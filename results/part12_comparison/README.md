# Four-method comparison: six-category macro means

Values are 0-1. DINO averages three seeds; the CSV includes sample-SD columns. Native/common scores and normal-data budgets are distinct; see ../../docs/PART1_PART2_ANALYSIS.md.

| Method | Shot | Scoring | Image AUROC | Image AP | Test FPR | Pixel AUROC | Pixel AP | AUPRO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | not_applicable | common_heatmap_top1pct | 0.6562 | 0.7661 | 0.0828 | 0.8131 | 0.1624 | 0.5543 |
| autoencoder | not_applicable | common_heatmap_top1pct | 0.7680 | 0.8408 | 0.0583 | 0.6941 | 0.0312 | 0.5580 |
| dino | 1 | common_heatmap_top1pct | 0.9440 | 0.9676 | 0.0750 | 0.9822 | 0.4387 | 0.9481 |
| dino | 1 | official_native | 0.9393 | 0.9612 | 0.0717 | 0.9822 | 0.4387 | 0.9481 |
| dino | 2 | common_heatmap_top1pct | 0.9473 | 0.9705 | 0.0683 | 0.9841 | 0.4566 | 0.9519 |
| dino | 2 | official_native | 0.9444 | 0.9687 | 0.0628 | 0.9841 | 0.4566 | 0.9519 |
| dino | 4 | common_heatmap_top1pct | 0.9579 | 0.9760 | 0.0704 | 0.9853 | 0.4749 | 0.9539 |
| dino | 4 | official_native | 0.9539 | 0.9737 | 0.0550 | 0.9853 | 0.4749 | 0.9539 |
| dino | 8 | common_heatmap_top1pct | 0.9665 | 0.9812 | 0.0800 | 0.9867 | 0.4898 | 0.9569 |
| dino | 8 | official_native | 0.9642 | 0.9798 | 0.0678 | 0.9867 | 0.4898 | 0.9569 |
| clip | 0 | common_heatmap_top1pct | 0.9026 | 0.9491 | 0.0467 | 0.9649 | 0.3126 | 0.9025 |
| clip | 0 | official_native | 0.8913 | 0.9353 | 0.1017 | 0.9649 | 0.3126 | 0.9025 |

## Fixed-threshold brightness study

| Method | Brightness factor | Pooled normal validation FPR |
| --- | --- | --- |
| autoencoder | 0.75 | 0.5093 |
| autoencoder | 1.0 | 0.0432 |
| autoencoder | 1.25 | 0.9969 |
| template | 0.75 | 0.1481 |
| template | 1.0 | 0.0432 |
| template | 1.25 | 0.4198 |

Scripts: illumination_course448.py, build_part12_comparison.py, verify_part12_supplement.py. Raw brightness predictions stay local. Thresholds are unchanged and the 1.0 condition reproduces original decisions. Brightening includes highlight clipping; no defect-recall claim is made.

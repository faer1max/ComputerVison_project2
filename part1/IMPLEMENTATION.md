# Part 1 implementation notes

The template baseline fits at most 24 normal references, registers them to a shared anchor and averages successful alignments. Testing uses Euclidean ECC, absolute RGB difference/255, Gaussian kernel 5x5 with sigma 0.8, inverse coordinate restoration and unaligned-border fallback. Failed registration retains the image. This is a documented mean-template variation rather than per-test nearest-template selection.

The compact AE uses four convolutional downsampling stages and symmetric transposed-convolution decoding at 448x448. Fit rows are normal only; validation rows are disjoint. Training uses MSE/Adam, 10 epochs, batch 16, lr 0.001 and seed 42, selecting the lowest normal-validation MSE. RGB uint8 fit/validation images are cached once; timing excludes cache preparation. The anomaly map is channel-mean absolute reconstruction error.

Both methods preserve all evaluated pixels. Scores, masks, threshold calibration and AUPRO are shared. Tests cover translation/rotation coordinate restoration, borders, failed ECC, low-value binary masks, splits, threshold ties and perfect/constant AUPRO. `verify_part1_release.py` verifies complete outputs and recomputed image metrics. See the part README for commands, and the analysis document for observed failures and limits.

# Part 2 architecture, directory and generated-output report

The complete course448 experiment contains 72 DINO and six CLIP configurations, verified against 16,692 finite 448x448 float32 maps. These are course-adapted shared-input results, not a claim to duplicate each paper's original resolution/settings.

## Model architecture

**AnomalyDINO:** frozen DINOv2 ViT-S/14 (12 layers, patch width 14, feature width 384) maps a 448 RGB input to 32x32 = 1,024 patch features. Each normal reference is rotated through eight angles. L2-normalized reference features populate CPU FAISS IndexFlatL2. Test foreground is selected by the official PCA strategy; reference masking is disabled. Squared L2/2 on normalized features is cosine-equivalent distance. Official interpolation/smoothing returns a 448 map. No target-category gradient training occurs. Shots 1/2/4/8 use nested random prefixes from normal fit rows only.

**AnomalyCLIP:** frozen CLIP ViT-L/14@336px uses the shared 448 input and official positional interpolation. The vision transformer has 24 layers and width 1,024, projected into the 768-dimensional shared space; text has 12 transformer layers, width 768 and a 49,408-token vocabulary. MVTec-trained prompt parameters use depth9/n_ctx12/t_n_ctx4, epoch15. The runner uses features24, DAPM_replace(DPAM_layer=20), and sigma4 Gaussian smoothing. The parameter 20 does not mean that only layer20 is replaced. Global image/text probability is the native image score; patch/text similarity yields the local map. No VisA prompt/backbone optimization occurs. Normal validation still calibrates the operational threshold.

Weight-derived dimensions are in `model_architecture.json`. Official source revisions and weight identities are in `assets/source_revisions.json` and `assets_verified.json`.

## Code and assets

```text
experiment_protocol.json       shared geometry/splits/threshold/metric contract
part1/common.py                VisA loading, normal split and common image scoring
part1/03_evaluate.py           shared pixel metrics and AUPRO
part2/
  benchmark.py                 model loading, caches, complete inference/evaluation
  course_inputs.py             shared RGB and model-specific normalization
  reference_selections.csv     270 uses / 72 configurations, with nested reuse
  check_contract.py            split, RNG, shared RGB and normalization-inverse checks
  verify_results.py            complete row/map/score/reference/source verification
  plot_results.py              few-shot curves and six-category panels
  export_failures.py           fixed-operating-point inventory and selected panels
  requirements-deploy.txt      isolated compatibility package versions
  faiss-win64-explicit.txt     Windows CPU FAISS sidecar package lock
  environment_verified.json   measured Python/torch/CUDA/GPU information
  model_architecture.json     architecture shapes read from official weights
  *_equivalence.json          official-entry sample numerical comparisons
external/                      pinned official clones, excluded from Git
weights/                       official backbone files, excluded from Git
data/VisA/                     raw official images/masks/split, excluded from Git
```

Use PREPARATION for rebuilding excluded assets. Official implementations are [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Shapes and flow

RGB ndarray `[448,448,3]` means height/width/color channels. Model tensor `[1,3,448,448]` means batch/channels/height/width. DINO features `[1024,384]` mean patches/feature dimension. An 8-shot, eight-rotation bank contains up to 65,536 rows of 384 float32 values. The final `[448,448]` map contains continuous pixel scores. Raw VisA masks use nearest resizing and `>0`; thresholding at 127 would lose low-valued foreground annotations.

DINO native scoring takes the highest ten of 1,024 patches (official floor). Common heatmap scoring takes the highest 2,008 of 448x448 pixels (ceil). The same nominal percentage is not the same aggregation. CLIP native probability is different again; all are explicitly named.

## Full output layout

```text
<CVPROJECT_PART2_OUTPUT>/
  smoke/                                   interface checks only
  full/
    source.json                            actual historical run identity
    cache/dino/<category>/*.npy             rebuildable frozen features
    dino/seed_<42|43|44>/shot_<1|2|4|8>/<category>/
      run.json                             references, bank size, timings, GPU peak
      metrics.json                         common image scoring and pixel metrics
      validation/predictions.csv           normal validation scores/maps index
      validation/maps/*.npy                 raw validation maps
      test/predictions.csv                 complete official test index
      test/maps/*.npy                       raw test maps
    clip/<category>/                       same output structure without shot levels
results/part2_course448/                    shareable CSV/JSON/PNG summaries
```

Each six-category pass contains 324 normal validation and 960 test images. Twelve DINO passes plus one CLIP pass yield 16,692 maps. The 270 reference-use records include repeated nested uses, not 270 unique images. Part 2 does not train new model checkpoints or training-loss histories: downloaded backbones/prompts are inputs, memory-bank/cache arrays are intermediates, and prediction maps/metrics are outputs.

Shareable files include `all_runs.csv` (78 rows), per-category and macro DINO mean/SD tables, six CLIP category rows, `run_metadata.json`, `efficiency.csv`, source verification, two-group pixel recomputation evidence, array inventory, few-shot curves and representative/failure panels. The full raw maps, caches and absolute-path prediction CSVs stay local.

## Numerical summaries

Values below are 0-1. DINO first averages six categories within each seed, then reports the mean and sample SD (ddof=1) over three seeds. CLIP is deterministic zero-shot and has no random-reference SD.

| Method | Native image AUROC | Native image AP | Pixel AUROC | Pixel AP | AUPRO | Native test FPR |
| --- | --- | --- | --- | --- | --- | --- |
| DINO 1-shot | 0.9393 +/- 0.0138 | 0.9612 +/- 0.0033 | 0.9822 +/- 0.0010 | 0.4387 +/- 0.0071 | 0.9481 +/- 0.0037 | 0.0717 +/- 0.0145 |
| DINO 2-shot | 0.9444 +/- 0.0032 | 0.9687 +/- 0.0019 | 0.9841 +/- 0.0010 | 0.4566 +/- 0.0121 | 0.9519 +/- 0.0003 | 0.0628 +/- 0.0175 |
| DINO 4-shot | 0.9539 +/- 0.0041 | 0.9737 +/- 0.0028 | 0.9853 +/- 0.0006 | 0.4749 +/- 0.0087 | 0.9539 +/- 0.0015 | 0.0550 +/- 0.0083 |
| DINO 8-shot | 0.9642 +/- 0.0039 | 0.9798 +/- 0.0026 | 0.9867 +/- 0.0002 | 0.4898 +/- 0.0100 | 0.9569 +/- 0.0010 | 0.0678 +/- 0.0025 |
| CLIP zero-shot | 0.8913 | 0.9353 | 0.9649 | 0.3126 | 0.9025 | 0.1017 |

| CLIP category | native_image_auroc | native_image_ap | pixel_auroc | pixel_ap | aupro | native_test_fpr |
| --- | --- | --- | --- | --- | --- | --- |
| candle | 0.8654 | 0.8719 | 0.9890 | 0.3024 | 0.9582 | 0.1000 |
| capsules | 0.8620 | 0.9113 | 0.9532 | 0.2893 | 0.8141 | 0.0500 |
| cashew | 0.7980 | 0.9108 | 0.9308 | 0.1827 | 0.9198 | 0.1200 |
| chewinggum | 0.9680 | 0.9862 | 0.9920 | 0.5677 | 0.8994 | 0.0200 |
| fryum | 0.9358 | 0.9713 | 0.9424 | 0.2212 | 0.8518 | 0.1400 |
| pipe_fryum | 0.9184 | 0.9600 | 0.9822 | 0.3124 | 0.9718 | 0.1800 |

Common heatmap image-score comparison:

| Method | image_auroc | image_ap | test_fpr |
| --- | --- | --- | --- |
| DINO 1-shot | 0.9440 +/- 0.0118 | 0.9676 +/- 0.0051 | 0.0750 +/- 0.0088 |
| DINO 2-shot | 0.9473 +/- 0.0043 | 0.9705 +/- 0.0022 | 0.0683 +/- 0.0142 |
| DINO 4-shot | 0.9579 +/- 0.0052 | 0.9760 +/- 0.0031 | 0.0704 +/- 0.0128 |
| DINO 8-shot | 0.9665 +/- 0.0040 | 0.9812 +/- 0.0025 | 0.0800 +/- 0.0176 |
| CLIP zero-shot | 0.9026 | 0.9491 | 0.0467 |

Thresholds use the normal-validation higher quantile and strict greater-than. Validation target FPR is not a guarantee that test FPR is at most 5%. Individual category/seed results need not be monotonic in shot count; references and parameters were not changed to enforce monotonicity.

## Files to retain and timing limits

| Artifact | Files | Bytes | GiB |
| --- | --- | --- | --- |
| prediction_maps | 16692 | 13402741248 | 12.482 |
| feature_cache | 1427 | 3819096448 | 3.557 |

A DINO reference occupies eight rotations x 1,024 patches x 384 dimensions x four bytes. Banks for 1/2/4/8 shots are 12/24/48/96 MiB, excluding Python/FAISS overhead. CLIP has no normal-reference memory bank. Keep source/protocol/reference lists and shareable summaries in Git. Keep full predictions locally for pixel recomputation; feature caches can be rebuilt; environments and weights are not submitted as generated results.

DINO extraction timers are cumulative only within the current process/category and reset after interruption; cache hits add no new extraction. Do not sum per-run cumulative records or call the final resumed timer a complete cold-start cost. Bank build time includes cache loading/concatenation/normalization/index insertion. Search timing includes nearest-neighbor search and map generation, excluding PCA foreground preparation, query-cache IO, model loading, metrics and plotting. CLIP separates measured feature extraction from text/patch scoring and CPU smoothing, excluding image loading/conversion from GPU extraction time. Peaks are PyTorch allocated memory, not total process memory. Some stages ran concurrently; no isolated end-to-end FPS ranking is claimed.

## Verification and failures

Shared RGB/splits and exact random references passed contract checks. Official-entry sample heatmap differences: CLIP 12 images, max absolute error 0; DINO six matched images, about 4.63e-8. These sample checks do not substitute for the full 78-group/16,692-map verification. Two complete candle groups separately recomputed pixel AUROC/AP/AUPRO and pixel calibration with zero difference.

The failure inventory covers only DINO seed42/1-shot and CLIP native operating points: DINO 24 FP/155 FN; CLIP 36 FP/181 FN. Twenty-four displayed method-image cases may repeat a product image across methods. They are grouped by observable false alarms or annotated defect extent, not proven causes. Panels have been visually reviewed; broader image/GT response, oversegmentation, small-defect misses and image/pixel disagreements are retained honestly. Do not tune Part 2 on these test examples.

For English-translated source compatibility, see ../docs/SOURCE_PROVENANCE.md. Historical hashes retain their actual execution meaning. For commands see README.md; for cross-method interpretation see ../docs/PART1_PART2_ANALYSIS.md. Part 3 remains pending.

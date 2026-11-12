# Part 2: AnomalyDINO and AnomalyCLIP on course448

Complete [environment and large-asset preparation](../PREPARATION.md) first. The provided runner requires Windows, CUDA torch/torchvision, official pinned clones, raw VisA with its split CSV, DINOv2 and CLIP backbone weights, and the MVTec-trained AnomalyCLIP prompt checkpoint.

## Install isolated compatibility dependencies

```powershell
python -m pip install --target part2/.deps --no-deps -r part2/requirements-deploy.txt
conda create --prefix ./part2/.faiss --file part2/faiss-win64-explicit.txt -y
New-Item -ItemType Directory -Force part2/.faiss/bin | Out-Null
$env:CVPROJECT_ASSETS=(Get-Location).Path
$env:CVPROJECT_DATA_ROOT=(Join-Path (Get-Location) 'data/VisA')
$env:CVPROJECT_PART2_OUTPUT=(Join-Path (Get-Location) 'outputs/course448/part2')
```

The process imports NumPy 1.26.4 and pandas 2.2.3 from `.deps`; it does not replace the base Part 1 NumPy 2 environment. FAISS is the pinned conda CPU build using exact `IndexFlatL2`. Do not use `KMP_DUPLICATE_LIB_OK` to suppress OpenMP conflicts. The explicit lock file is win-64 only. Its package URLs require network access; a pre-existing compatible asset directory can instead be reused with `CVPROJECT_ASSETS`.

## Smoke test, then full experiments

```powershell
python part2/benchmark.py dino --smoke
python part2/benchmark.py clip --smoke
python part2/check_contract.py
python part2/benchmark.py dino
python part2/benchmark.py clip
python part2/verify_results.py
python part2/plot_results.py
python part2/export_failures.py
```

Smoke outputs live under `smoke/`; they never substitute for `full/`. Full completion requires 72 DINO groups (six categories x four shots x three seeds) and six CLIP groups, or 16,692 validation/test maps. Each configuration contains complete validation/test prediction CSVs, float32 448x448 maps, `run.json` and `metrics.json`. The frozen DINO cache is rebuildable. Already completed groups can resume only with matching source identity or the explicit audited translation bridge; unrelated code changes require a new output root. Historical `source.json` is never overwritten just because a translated verifier recognizes it.

DINO uses seeds 42/43/44, normal-fit-only nested references, eight rotations, official PCA test masking and CPU cosine-equivalent nearest-neighbor search. CLIP loads `checkpoints/9_12_4_multiscale_visa/epoch_15.pth`, trained on MVTec, with depth 9, n_ctx 12, t_n_ctx 4, features 24, DPAM parameter 20 and sigma 4. No VisA gradient training occurs. Normal validation still calibrates the operating threshold.

Image native scores (DINO patch top-1%, CLIP global image/text probability) are separate from common heatmap top-1% scores. Curves report means and sample SD over three DINO seeds, first averaging six categories within each seed. Do not confuse either score with pixel metrics. Model normalization differs by pretrained interface while RGB geometry and evaluation remain shared.

See [architecture/output report](REPORT.md), [analysis](../docs/PART1_PART2_ANALYSIS.md), and [verification checklist](../docs/PART1_PART2_REQUIREMENT_CHECKLIST.md). Published summaries exclude arrays, caches and weights. Partial concurrent execution, cache hits and restarted extraction timers limit the interpretation of resource times; no isolated end-to-end speed ranking is claimed.

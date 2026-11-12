# Preparing the environment and excluded large files

Run commands from the repository root in **Windows PowerShell**. The Part 2 bootstrap and FAISS lock file are Windows-specific; Linux/macOS require a separate FAISS/DLL adapter and have not been validated. Allow at least 50 GB of free working space for datasets, environments, caches and full results. Published summaries are much smaller.

## 1. Clone this project and create the Python environment

```powershell
git clone https://github.com/faer1max/ComputerVison_project2.git
cd ComputerVison_project2
conda create -n cvproject python=3.11 -y
conda activate cvproject
```

If `cvproject` already exists, activate and inspect it instead of recreating or overwriting it. Tested hardware: RTX 4060 Laptop GPU (8 GB); Python 3.11.17; torch 2.12.0+cu130; torchvision 0.27.0+cu130. These are recorded versions, not a claim that they are the latest or available for every driver. Install a mutually compatible CUDA torch/torchvision pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/). For the recorded CUDA 13.0 build, when available for your platform:

```powershell
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip install -r part1/requirements.txt
```

Part 2 requires CUDA in the provided runner. Stop and resolve incompatible GPU/driver/torchvision builds before a full experiment. Part 1 supports CPU, but the published full runs used the recorded GPU environment. Exact historical environment information is under `results/part1_course448/environment.*` and `part2/environment_verified.json`.

## 2. Obtain the four official source repositories

```powershell
python scripts/prepare_repositories.py --install
python scripts/prepare_repositories.py
```

The installer clones missing repositories and checks out the exact commits in `assets/source_revisions.json`. It rejects an existing incompatible revision or modified tracked Python code instead of overwriting it. The prompt checkpoint is supplied by the pinned AnomalyCLIP clone. Git must be installed and GitHub must be reachable.

Equivalent commands for a clean `external/` directory:

```powershell
git clone --no-checkout https://github.com/dammsi/AnomalyDINO.git external/AnomalyDINO
git -C external/AnomalyDINO checkout --detach b9d1c2648e3a5247437d4d953d907a8f3d994457
git clone --no-checkout https://github.com/zqhang/AnomalyCLIP.git external/AnomalyCLIP
git -C external/AnomalyCLIP checkout --detach 3911738c0867544f545a076ad78f3f11d9ecbfdf
git clone --no-checkout https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
git clone --no-checkout https://github.com/amazon-science/spot-diff.git external/spot-diff
git -C external/spot-diff checkout --detach 2a692ab575001cbde74d402d897a7286086c6199
```

## 3. Download excluded data and backbone weights

| Asset | Expected local path | Official source / reconstruction |
| --- | --- | --- |
| Raw VisA archive (1,929,840,640 bytes) | `data/downloads/VisA_20220922.tar` | [AWS archive](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) |
| DINOv2 ViT-S/14 backbone | `weights/dinov2_vits14_pretrain.pth` | [Meta weight file](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |
| CLIP ViT-L/14@336px backbone | `weights/ViT-L-14-336px.pt` | [OpenAI weight file](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt) |
| MVTec-trained AnomalyCLIP prompt | `external/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth` | Included in pinned AnomalyCLIP clone; not trained on VisA |
| Optional CVPR template | `assets/templates/CVPR2026-v1-latex.zip` | [CVPR author-kit release](https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)); not needed for model execution |

Download only the model/data assets needed for Parts 1/2:

```powershell
python scripts/prepare_assets.py --only "VisA"
python scripts/prepare_assets.py --only "DINOv2 ViT-S/14"
python scripts/prepare_assets.py --only "CLIP ViT-L/14@336px"
```

The script uses Windows `curl.exe`, validated HTTP byte ranges, resumable parts and safe archive extraction. Retry the same command after interruption; do not delete valid parts unnecessarily. A proxy/firewall must allow GitHub, the AWS archive, Meta's public files, the CLIP asset host and the conda/pip package hosts. Downloading from arbitrary mirrors is not part of the verified setup.

The script writes `assets/preparation_status.json`. CLIP's SHA256 is encoded in its official URL. DINO/VisA SHA256 values in the inventory identify the tested downloads; they are **not independently published official checksum attestations**. Model asset identities are also recorded in `part2/assets_verified.json`. A completed download alone does not prove inference works.

## Safe fallback when the upstream DINO TLS endpoint is unavailable

On 2026-11-12 the official DINO file host returned an expired-certificate error in both Python and Windows curl from the release environment. VisA and CLIP returned valid byte-range responses. Do not use `--insecure`, disable certificate checks or silently download a different model. The original official URL above remains the primary source when its TLS service is valid.

For an immediate handoff, obtain the **already verified original DINO checkpoint** from the project owner through an agreed file-transfer method (the owner's local file is `weights/dinov2_vits14_pretrain.pth`). It is intentionally not committed to Git. Validate and stage that exact binary:

```powershell
python scripts/stage_local_weights.py --asset dino --source "E:/handoff/dinov2_vits14_pretrain.pth"
python scripts/stage_local_weights.py --asset dino --verify-only
# Optional offline CLIP handoff, checked against the hash in its official URL:
python scripts/stage_local_weights.py --asset clip --source "E:/handoff/ViT-L-14-336px.pt"
```

The script refuses mismatched source or existing destination files, verifies a copied temporary file and atomically stages it under `weights/`. DINO expected SHA256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`; this is the previously verified project artifact identity, not an independently published official checksum. Do not rename a Hugging Face-format checkpoint into the torch-hub filename: different key layouts/serializations are not a verified replacement. An official Meta mirror was investigated but a complete compatible binary download was not verified in this environment, so no untested mirror-conversion command is provided.

A new recipient needs either restored upstream TLS availability or access to the verified handoff file. The Git clone alone cannot overcome an unavailable upstream service. After staging the file, skip the failed DINO download command and proceed normally. Raw data, AE checkpoints, caches and heatmaps are still prepared by the commands above/below.

## 4. Prepare official VisA splits and metadata

```powershell
python scripts/prepare_visa.py
```

This verifies all 10,821 raw images and annotated masks, invokes pinned `spot-diff/utils/prepare_data.py`, checks normal-only training/binary prepared masks, copies the official split to `data/VisA/split_csv/1cls.csv`, and writes `data/VisA/meta.json` for CLIP-compatible metadata. Prepared 1-class data are under `data/VisA_pytorch/1cls`. Keep normal raw masks absent; anomalous raw masks are binarized with `>0`, not `>127`.

The course runners read the raw `data/VisA` images with the official split, not a second resized/cropped prepared dataset. The data inventory under `assets/` is overwritten only by intentional preparation; historical published inventories describe the tested data, not guaranteed files on a new machine.

## 5. Run Part 1, then Part 2

Follow [Part 1 commands](part1/README.md) to create normal templates and AE checkpoints and calibrate/evaluate them. Then follow [Part 2 commands](part2/README.md) for isolated NumPy/FAISS dependencies, both smoke tests, the full few-shot/zero-shot runs and verification. Do not install the Part 2 NumPy 1 overlay over the base Part 1 environment. The provided FAISS lock is Windows win-64 only.

For the supplementary comparison, both parts' full outputs must already exist:

```powershell
python scripts/illumination_course448.py
python scripts/build_part12_comparison.py
python scripts/verify_part12_supplement.py
```

## Existing assets and output recovery

- `CVPROJECT_ASSETS`: directory containing `external/`, `weights/`, `part2/.deps/`, and `part2/.faiss/` for the Part 2 runner and repository checker.
- `CVPROJECT_DATA_ROOT`: raw VisA directory containing `split_csv/1cls.csv`.
- `CVPROJECT_OUTPUT_ROOT`: existing or new Part 1 output root.
- `CVPROJECT_PART2_OUTPUT`: Part 2 root containing `smoke/` and `full/`.

The download/data-preparation scripts prepare files under their own repository root. For a brand-new setup, leave defaults and run everything in one clone. If reusing assets elsewhere, set the runtime variables above; do not assume all preparation scripts relocate with the asset variable.

No pretrained project AE checkpoint is offered for download: regenerate it by normal-only training. Templates are regenerated by the template runner. Full heatmaps and DINO feature caches are regenerated by inference; they are not required to read the published CSV/PNG summaries. Saved prediction CSVs contain local absolute map paths: after moving an output directory to another machine, regenerate predictions into the new output root instead of blindly reusing invalid absolute paths. Keep historical output folders intact while testing a new setup.

## Troubleshooting and handoff boundaries

If imports fail, confirm the active interpreter with `python -c "import sys; print(sys.executable)"`. If FAISS raises an OpenMP conflict, use the pinned conda sidecar, not duplicate-runtime suppression. If source identity fails, use a new output root for actual code changes; only the explicit audited translation mapping may recognize historical source hashes. If models or templates are missing, run their generating commands before demos/evaluation.

AnomalyCLIP folder names can be misleading: the `_visa` prompt directory is the MVTec-trained checkpoint used to evaluate VisA. Do not substitute the other checkpoint trained on VisA. Preserve the official source revisions, weights, random-reference manifest and protocol when reproducing results.

VisA images/annotations originate from [SPot-the-Difference](https://github.com/amazon-science/spot-diff) (CC BY 4.0). Cite VisA, DINOv2, CLIP, AnomalyDINO, AnomalyCLIP and other borrowed components; retain upstream license obligations. Do not redistribute course PDFs or third-party checkpoints merely because they were available locally.

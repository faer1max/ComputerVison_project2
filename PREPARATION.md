# 开工准备

## 文件位置

| 内容 | 路径 |
| --- | --- |
| VisA 官方完整压缩包 | `data/downloads/VisA_20220922.tar` |
| VisA 原始数据及 AnomalyCLIP 元数据 | `data/VisA/`、`data/VisA/meta.json` |
| 官方 1-class 数据目录 | `data/VisA_pytorch/1cls/` |
| AnomalyDINO 官方源码 | `external/AnomalyDINO/` |
| AnomalyCLIP 官方源码及提示学习检查点 | `external/AnomalyCLIP/` |
| DINOv2 官方源码 | `external/dinov2/` |
| VisA 官方划分与预处理代码 | `external/spot-diff/` |
| DINOv2 ViT-S/14 骨干权重 | `weights/dinov2_vits14_pretrain.pth` |
| CLIP ViT-L/14@336px 骨干权重 | `weights/ViT-L-14-336px.pt` |
| 课程 PDF 指定的 CVPR 模板 | `assets/templates/CVPR2026-v1-latex.zip` |

下载完成情况以 `assets/preparation_status.json` 为准，数据检查结果见
`assets/visa_inventory.json`（数据准备完成后生成）。

官方仓库 URL 和精确 commit 保存在 `assets/source_revisions.json`。`external/`、数据、模型权重与模板压缩包不纳入本项目 Git 提交。

## 复现准备过程

在项目根目录执行（Python 3.11（本次实测3.11.17），系统有 curl；数据整理另需 numpy、Pillow）：

```powershell
python scripts/prepare_repositories.py --install
python scripts/prepare_assets.py
python scripts/prepare_visa.py
```

下载脚本使用官方 URL，分段下载支持重跑复用已完整下载的分段。记录每个完整文件的 SHA-256；CLIP 另外与官方 URL 中的 SHA-256 核对。DINOv2 和 VisA 的本地 SHA-256 用于记录文件身份，不代表已有官方校验值。VisA 解压采用安全路径过滤，随后检查原图和标注可解码、官方划分无重复图、训练集全为正常图、整理后的异常掩码已二值化。

`prepare_visa.py` 调用未修改的官方 `external/spot-diff/utils/prepare_data.py` 整理数据；另按 AnomalyCLIP 官方字段约定生成原始数据目录中的 `meta.json`，不改动官方 train/test 划分。数据目录中 `pipe_fryum` 对应课程说明中的 pipe fryum。

## 模型使用注意

- AnomalyDINO 官方默认骨干是 `dinov2_vits14`，对应已准备的 ViT-S/14 权重。
- 在 VisA 上评估 AnomalyCLIP 应使用 `external/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth`。官方 `train.sh` 明确将这组模型训练在 MVTec 上；没有 `_visa` 后缀的另一组模型训练在 VisA 上，不用于这里的零样本 VisA 评估。不能只根据脚本注释或目录名猜测训练来源。
- 骨干权重已放在项目 `weights/` 中，后续运行时需要明确配置本地加载/缓存路径，避免默认重新下载到 C 盘。
- 文件完整性检查不等于模型推理验证。本分支现已完成Part1与Part2的正式实验；Part2模型加载、官方接口数值核对、指标及显存记录见part2/REPORT.md和results/part2_course448。

## 运行环境与待办

2026-11-05：Part 1 使用 Python 3.11 / CUDA PyTorch；共享输入协议固定于
`experiment_protocol.json`，精确运行环境和验证摘要见 `results/part1_course448`。
数据默认位于项目内 `data/VisA`；通过 `CVPROJECT_DATA_ROOT` 可以复用其他目录的数据。
运行命令以 `part1/README.md` 为准。Part 2 的FAISS/NumPy依赖已采用独立配置，复现命令见part2/README.md，
不要用Part1的requirements直接覆盖其环境。数据、权重和完整实验输出不随Git提交。

## 官方来源

- VisA：<https://github.com/amazon-science/spot-diff>
- AnomalyDINO：<https://github.com/dammsi/AnomalyDINO>
- AnomalyCLIP：<https://github.com/zqhang/AnomalyCLIP>
- DINOv2：<https://github.com/facebookresearch/dinov2>
- CVPR 模板：<https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)>

## 从空目录准备官方源码（固定版本）

先安装Git以及Python3.11环境和匹配的CUDA torch/torchvision，再安装part1/requirements.txt。默认资产放仓库根目录；--install仅克隆缺失仓库，已有版本不符或有Python改动会拒绝覆盖。可复用资产时设置CVPROJECT_ASSETS；数据准备脚本仍从其所在仓库根目录准备数据，跨目录复用数据应直接设置CVPROJECT_DATA_ROOT而不重复整理。

```powershell
python scripts/prepare_repositories.py --install
# 只核对已有源码，不下载：
python scripts/prepare_repositories.py
```

等效手工命令（在仓库根目录执行，仅用于尚未克隆时）：

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

顺序：官方源码 → prepare_assets下载并解压VisA/骨干 → prepare_visa官方划分与CLIP元数据 → Part1正式训练/评估 → Part2独立依赖与推理。AnomalyCLIP提示checkpoint随固定官方仓库提供；不得换成VisA训练的另一组提示。已有数据/权重可直接复用，不需清空或重装cvproject。

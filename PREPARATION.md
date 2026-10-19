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

在项目根目录执行（Python 3.12+，系统有 curl；数据整理另需 numpy、Pillow）：

```powershell
python scripts/prepare_assets.py
python scripts/prepare_visa.py
```

下载脚本使用官方 URL，分段下载支持重跑复用已完整下载的分段。记录每个完整文件的 SHA-256；CLIP 另外与官方 URL 中的 SHA-256 核对。DINOv2 和 VisA 的本地 SHA-256 用于记录文件身份，不代表已有官方校验值。VisA 解压采用安全路径过滤，随后检查原图和标注可解码、官方划分无重复图、训练集全为正常图、整理后的异常掩码已二值化。

`prepare_visa.py` 调用未修改的官方 `external/spot-diff/utils/prepare_data.py` 整理数据；另按 AnomalyCLIP 官方字段约定生成原始数据目录中的 `meta.json`，不改动官方 train/test 划分。数据目录中 `pipe_fryum` 对应课程说明中的 pipe fryum。

## 模型使用注意

- AnomalyDINO 官方默认骨干是 `dinov2_vits14`，对应已准备的 ViT-S/14 权重。
- 在 VisA 上评估 AnomalyCLIP 应使用 `external/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth`。官方 `train.sh` 明确将这组模型训练在 MVTec 上；没有 `_visa` 后缀的另一组模型训练在 VisA 上，不用于这里的零样本 VisA 评估。不能只根据脚本注释或目录名猜测训练来源。
- 骨干权重已放在项目 `weights/` 中，后续运行时需要明确配置本地加载/缓存路径，避免默认重新下载到 C 盘。
- 文件完整性检查不等于模型推理验证；尚未进行模型加载、训练、指标评估或显存测试。

## 运行环境与待办

2026-10-18 检查：本机 NVIDIA RTX 4060 Laptop GPU，约 8 GB 显存；D 盘准备前约 95 GiB 可用，C 盘约 7 GiB 可用。环境和缓存优先放在 D 盘。

现有 `D:/conda/envs/cv` 是 Python 3.11，但尚未安装 PyTorch。本次准备不修改该环境。后续需配置独立、兼容本机 CUDA 驱动的 PyTorch 环境，安装图像处理、评估及检索依赖，再先运行单张图验证。两份官方 requirements 的年代和版本约束不同，不能直接假定可以混装；Windows 上的 FAISS 安装方案也需验证。

开始实现无需用户另行上传数据。若课程还有额外评分细则、截止日期、老师补充说明或已有实验记录，可补充到项目中。作者、队友及最终提交账号信息可在报告与提交阶段提供。

## 官方来源

- VisA：<https://github.com/amazon-science/spot-diff>
- AnomalyDINO：<https://github.com/dammsi/AnomalyDINO>
- AnomalyCLIP：<https://github.com/zqhang/AnomalyCLIP>
- DINOv2：<https://github.com/facebookresearch/dinov2>
- CVPR 模板：<https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)>

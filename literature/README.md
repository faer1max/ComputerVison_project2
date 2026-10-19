# 推荐文献库

对应 Project2 第 4 节的全部十篇推荐阅读，按课程原有四类整理。

| 分类 | 论文 | 下载版本 | 官方来源 |
| --- | --- | --- | --- |
| 基础方法 | [MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](01_foundations/01_MVTec_AD_CVPR2019.pdf) | CVPR 2019 | [PDF 来源](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf) |
| 基础方法 | [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](01_foundations/02_PaDiM_ICPR2021_arXiv.pdf) | arXiv v1; ICPR workshop paper | [PDF 来源](https://arxiv.org/pdf/2011.08785v1) |
| 基础方法 | [Towards Total Recall in Industrial Anomaly Detection](01_foundations/03_PatchCore_CVPR2022.pdf) | CVPR 2022 | [PDF 来源](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) |
| 重建与合成 | [DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection](02_reconstruction_and_synthesis/04_DRAEM_ICCV2021.pdf) | ICCV 2021 | [PDF 来源](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) |
| 重建与合成 | [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](02_reconstruction_and_synthesis/05_CutPaste_CVPR2021.pdf) | CVPR 2021 | [PDF 来源](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf) |
| 基础模型 | [DINOv2: Learning Robust Visual Features without Supervision](03_foundation_models/06_DINOv2_TMLR2024_arXiv.pdf) | arXiv v2; TMLR 2024 paper | [PDF 来源](https://arxiv.org/pdf/2304.07193v2) |
| 基础模型 | [AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection](03_foundation_models/07_AnomalyCLIP_ICLR2024.pdf) | ICLR 2024 会议发表版 | [PDF 来源](https://openreview.net/notes/edits/attachment?id=HM1Swt7ABY&name=pdf) |
| 基础模型 | [AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2](03_foundation_models/08_AnomalyDINO_WACV2025.pdf) | WACV 2025 | [PDF 来源](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf) |
| 效率与鲁棒性 | [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](04_efficiency_and_robustness/09_EfficientAD_WACV2024.pdf) | WACV 2024 | [PDF 来源](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf) |
| 效率与鲁棒性 | [The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection](04_efficiency_and_robustness/10_MVTec_AD_2_IJCV2026.pdf) | IJCV 2026 正式开放获取版 | [PDF 来源](https://link.springer.com/content/pdf/10.1007/s11263-026-02743-0.pdf) |

## 文件说明

- `sources.json`：下载来源、版本说明与分类。
- `download_manifest.json`：每篇论文下载状态、页数、大小、SHA-256 和检查时间。
- `notes/`：后续共读笔记与提升思路，当前尚未开始精读。
- 根目录 `scripts/download_literature.py`：下载和核验脚本，需要 Python、pypdf 和 curl；已存在的 PDF 会先验证。

PDF 使用原始公开版本，未作翻译或修改。PaDiM 与 DINOv2 使用表中标明的 arXiv 版本；AnomalyCLIP 使用 ICLR 2024 会议版，MVTec AD 2 使用 IJCV 2026 期刊正式版。文件准备与身份核验已经完成，尚未进行逐篇精读。

## 建议共读方式

先浏览 MVTec AD 的问题与评估定义，再以 PatchCore → DINOv2 → AnomalyDINO → AnomalyCLIP 建立方法主线。其余论文结合 Part 1 基线和 Part 3 改进方向深入阅读。

每篇笔记记录：研究问题、方法流程、关键公式、实验依据、局限、与本项目的联系、待验证假设。区分论文作者结论与我们的推测。

# 四方法统一比较（六类宏平均）

所有数值为0–1。DINO为三种子均值；样本SD在CSV的_std列。common与native分开，参考预算和计时边界见../../docs/PART1_PART2_ANALYSIS.md。

| 方法 | shot | 评分 | image AUROC | image AP | test FPR | pixel AUROC | pixel AP | AUPRO |
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

## 光照压力实验

| 方法 | 亮度因子 | 合并正常验证FPR |
| --- | --- | --- |
| autoencoder | 0.75 | 0.5093 |
| autoencoder | 1.0 | 0.0432 |
| autoencoder | 1.25 | 0.9969 |
| template | 0.75 | 0.1481 |
| template | 1.0 | 0.0432 |
| template | 1.25 | 0.4198 |

脚本：scripts/illumination_course448.py、scripts/build_part12_comparison.py、scripts/verify_part12_supplement.py。原始逐图亮度分数留在本地，不上传大数组。所有实验固定原阈值，×1.0复现原CSV评分和决策；高光裁剪局限见分析文档。

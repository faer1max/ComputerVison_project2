# Part 2：Windows运行与验收

本分支以已验收Part1为基础，统一协议在experiment_protocol.json，固定方案在PLAN.md，完整目录/模型/结果说明见REPORT.md。

先按根PREPARATION准备官方external源码、weights和VisA。原始仓库commit记录在assets/source_revisions.json；官方源码不修改、不重复提交。先在CUDA PyTorch环境安装Part1基础依赖，再把Part2兼容依赖隔离安装：

```powershell
python -m pip install --target part2/.deps --no-deps -r part2/requirements-deploy.txt
conda create --prefix ./part2/.faiss --file part2/faiss-win64-explicit.txt
$env:CVPROJECT_ASSETS=(Get-Location).Path
$env:CVPROJECT_DATA_ROOT=(Join-Path (Get-Location) 'data/VisA')
$env:CVPROJECT_PART2_OUTPUT=(Join-Path (Get-Location) 'outputs/course448/part2')
python part2/benchmark.py dino --smoke
python part2/benchmark.py clip --smoke
python part2/benchmark.py dino
python part2/benchmark.py clip
python part2/verify_results.py
python part2/plot_results.py
```

GPU必须支持当前PyTorch CUDA构建。只在当前进程加载.deps的NumPy1.26，不覆盖Part1 NumPy2环境。
Part2另外需要与PyTorch匹配的torchvision；本次为torch2.12.0+cu130和torchvision0.27.0+cu130。请按PyTorch官方安装器选择匹配的GPU构建，再安装Part1基础依赖与上述独立兼容依赖，不要把CPU版或不匹配的torchvision混入。
FAISS为官方conda1.8 CPU版（精确检索）；pip旧版与本机OpenMP冲突，不设置KMP_DUPLICATE_LIB_OK。
`CVPROJECT_ASSETS`可指向已有共享资产目录；输出另指定独立位置，不能用smoke代替full。
已有metrics.json的组支持恢复，但source.json身份不同会拒绝复用，必须新建输出目录。

原生评分：DINO使用官方mean_top1p，CLIP使用全局图文异常概率；统一热图top1%评分作为独立对照保存。
六类图像/像素指标沿用Part1同一评估模块。正常validation分数定阈值，test不用于调参。DINO三种子均值/样本标准差ddof1；先每seed平均六类再跨seed计算宏曲线。

verify_results必须通过78组/16692张图、完整路径集合、参考图隔离、热图有限值和数值重算。原生分数图像指标另由CSV重新计算。
模型核心数值与官方入口的匹配样本对比：CLIP12图max_abs_error=0，DINO6匹配图max_abs_error约4.63e-8；详情见equivalence JSON，不能把该检查误称为全量结果验收。

输入448是课程公平比较配置，不冒充论文原分辨率结果。冻结特征缓存的命中/提取成本、CPU记忆库及GPU分配峰值需分开解读。Part3仍未完成。

补充复现准备：首次运行请先按根PREPARATION.md执行 `python scripts/prepare_repositories.py --install`，再下载数据/骨干并整理数据。该脚本固定assets/source_revisions.json中的四个仓库版本，不覆盖已有源码修改。

2026-11-12补齐核查：见 ../docs/PART1_PART2_REQUIREMENT_CHECKLIST.md 和 PART1_PART2_ANALYSIS.md；统一表与局部图在results/part12_comparison，光照对照在results/part1_course448/illumination。

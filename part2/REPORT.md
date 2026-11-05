# Part 2：目录、模型架构与结果文件报告

本文件是结构说明。正式数值和验收状态以 results/part2_course448 的生成文件为准；没有生成verification.json前，不能把流程检查或部分组结果称为完整复现。

## 模型本身的架构

### AnomalyDINO

冻结的DINOv2 ViT-S/14（官方vit_small：12层、每块384维）将448×448 RGB图像分成32×32个14×14块，即1024个块。
每类正常参考图按8个角度旋转，提取正常块特征，构成CPU FAISS IndexFlatL2记忆库。
测试块先由官方PCA策略筛选前景，特征L2归一化后精确找最近正常块，平方L2距离/2对应余弦距离。
距离还原为32×32块图，再用官方dists2map放大至448×448并平滑。无目标类别梯度训练。
参考图来自正常fit，不能来自validation/test。1/2/4/8-shot每个种子使用嵌套的随机前缀。

### AnomalyCLIP

冻结的CLIP ViT-L/14@336px骨干接受统一448输入，按官方代码插值位置编码。
图像分支提取全局图像特征与第24层patch特征；官方DPAM调用参数为20，具体替换范围遵照其源码，不能误写成只替换第20层。
文本分支加载在MVTec训练的提示参数（depth9/n_ctx12/t_n_ctx4，epoch15），编码正常/异常提示。
全局图像与文本相似度给出官方图像异常概率；patch与文本相似度给出局部异常图，sigma4高斯平滑。
不在VisA上训练提示或骨干，属于目标数据零样本比较。

## 项目代码结构

```text
experiment_protocol.json             # 与Part1共享448、划分、阈值和评估约定
part1/common.py                       # 已修复的官方VisA读取、正常划分、图像分数
part1/03_evaluate.py                  # 同一套AUROC/AP/FPR和AUPRO计算
part2/
  PLAN.md                            # 本轮固定实验方案
  benchmark.py                       # 模型加载、特征缓存、72+6组推理、共享评估
  course_inputs.py                   # 共享RGB/bilinear448、模型接口归一化
  reference_selections.csv            # 270条参考图使用记录，72组随机配置
  verify_results.py                  # 输出、源码、参考图、热图及原生图像分数核验
  REPORT.md                          # 本报告
external/                            # 官方源码（不提交仓库，按固定commit准备）
  AnomalyDINO/src/backbones.py        # DINO特征提取与PCA前景策略
  AnomalyDINO/src/utils.py            # 官方旋转增强、距离图插值和平滑
  AnomalyCLIP/AnomalyCLIP_lib/        # CLIP双分支与DPAM实现
  AnomalyCLIP/prompt_ensemble.py      # 提示参数结构
  dinov2/                            # 官方DINOv2网络定义
weights/                             # 官方骨干权重，不提交
```

bootstrap显式读取项目内.deps和官方conda FAISS DLL，保留原cvproject环境。共享几何预处理完全相同；DINO和CLIP各自预训练均值/标准差属于模型接口，显式区分。
`load_eval()`返回已发布Part1评估模块；`evaluate(directory,category)`读取当前组validation/test CSV和热图，只用正常validation校准阈值，再计算指标。
`save_predictions(rows,maps,native,directory)`检查448形状及有限值，逐图保存float32数组，并输出带原生分数和统一热图分数的索引表。
`run_dino(categories,smoke)`从固定清单构建记忆库，缓存冻结特征，再执行检索；`run_clip(categories,smoke)`加载官方提示和骨干，进行图文推理。
这些是项目自定义函数；torch、FAISS、Pillow的调用是库API，不是可任意改名的接口。

## 数据形状与流向

RGB读取结果是uint8 ndarray `[448,448,3]`，三维为高、宽、颜色通道。
网络输入Tensor `[1,3,448,448]` 为批次、通道、高、宽。
DINO特征`[1024,384]`为图片块数、块特征维度。8-shot×8旋转时参考库最多65536行，每行384个float32数；前景只筛测试特征，参考mask关闭，保持官方默认。
最终热图`[448,448]`每个位置是一项连续异常分数。真实掩码用nearest/>0处理，不能用127切掉VisA的低值前景。

## 本地运行结果目录

```text
outputs/course448/part2_final/
  smoke/                             # 每类2正常验证+正常/异常测试各1张，不能作正式结果
  full/
    source.json                      # 实验源码、共享协议、参考清单、评估器SHA256
    cache/dino/<category>/*.npy       # 一次提取的冻结块特征，可重建
    dino/seed_<42|43|44>/shot_<1|2|4|8>/<category>/
      run.json                       # 参考路径、记忆库字节、构建/检索耗时、显存
      metrics.json                   # 当前组统一热图评分与像素指标
      validation/predictions.csv     # 正常验证索引，仅此组用于阈值
      validation/maps/*.npy          # 每张正常验证图的原始浮点热图
      test/predictions.csv           # 完整官方测试索引
      test/maps/*.npy                # 每张测试图的原始浮点热图
    clip/<category>/                 # 同样的run/metrics/validation/test结构，无shot
results/part2_course448/              # 生成后可分享，不含权重、特征缓存和全量热图
  all_runs.csv                       # 78组逐组指标，包含native与common评分
  dino_category_mean_std.csv          # 每类每shot三种子的均值/样本标准差
  dino_macro_mean_std.csv             # 每种子先平均六类，再跨种子统计
  clip_categories.csv                # 六类零样本结果
  verification.json                  # 完整性与源码身份验收
```

完整正式输出应为72组DINO+6组CLIP，共16692张热图：DINO12次完整六类validation/test预测，CLIP1次。
validation共324张正常图，test共960张。270条参考使用记录包含嵌套复用，不等于270张不同图片。
图像原生评分与统一热图top1%评分分别报告：前者用于官方评分复现曲线，后者是共享聚合对照；不能混着解释。
每类每组阈值均只由正常validation确定，test只在最后计算指标。AUPRO为8邻域、200阈值、FPR上限.30归一化。
缓存命中时不重新提取特征：run.json的累计提取时间和构库/检索时间不能简单加成无缓存速度；显存也应说明是否发生特征提取。

## 环境与运行

配置CVPROJECT_ASSETS指向已经按assets/source_revisions.json准备好external/weights和part2/.deps、.faiss的资产根目录；CVPROJECT_DATA_ROOT指向VisA；CVPROJECT_PART2_OUTPUT指定上述输出根。
使用cvproject的Python执行：

```powershell
python part2/benchmark.py dino --smoke
python part2/benchmark.py clip --smoke
python part2/benchmark.py dino
python part2/benchmark.py clip
python part2/verify_results.py
```

报告只描述本轮固定配置，不代表论文原生分辨率的原样数字；448是课程共同输入要求下的适配。
Part3仍需在共同协议下做有依据的改进和消融，不能因Part2完成而省略。

# Part 1：课程统一448基线

两个方法：最多24张正常图配准后的均值模板；正常图卷积自编码器。此版本修复了旧
common.py 语法错误和模板热图坐标错误，并对六类按新协议重新运行。
运行证据与指标见 `results/part1_course448/`；源码身份由每次运行的 SHA-256 记录。

## 环境与数据

Python 3.11；本次实测 PyTorch 2.12.0+cu130 / RTX 4060 Laptop 8GB。
按 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/) 安装适合本机的
PyTorch；精确复现本次GPU设置需要同版本PyTorch与驱动。CPU可运行但未做全量速度验证。
然后执行 `python -m pip install -r part1/requirements.txt`。
requirements 锁定本次实际使用的直接依赖，完整环境快照见结果目录的 environment.txt。
本分支的NumPy2环境与未来Part2需要NumPy1的隔离依赖分别配置，不能直接混装。

默认读取 `data/VisA/split_csv/1cls.csv` 和对应原始图像。可运行
`python scripts/prepare_assets.py`、`python scripts/prepare_visa.py`，数据与模型不上传。
数据已在其他目录时，用 PowerShell 设置 `$env:CVPROJECT_DATA_ROOT='你的VisA目录'`。
输出默认位于 `outputs/course448/part1/`；可用 `CVPROJECT_OUTPUT_ROOT` 指定输出目录。

## 唯一正式协议

根目录 `experiment_protocol.json` 固定输入和默认设置。六类均使用 RGB、Pillow双线性
缩放为448×448，不裁边、不填充；掩码最近邻缩放后以>0二值化。448同时兼容14patch
与16倍AE下采样。固定seed42，从官方正常train中留10%作validation，test保持官方划分。
fit/validation/test数量共2921/324/960；test含360正常、600异常。

模板使用拟合池前24张，配准成功者求均值。ECC为欧氏变换、80次/epsilon1e-5；
差异图为RGB平均绝对差/255，高斯核5×5、sigma0.8。输出热图映回原测试图坐标；
无法映射的边界使用未配准差异回退，配准失败同样回退，所有测试像素均参与指标。
预测CSV记录registration_ok及mapped_pixel_fraction，不删除失败样本。

AE四级卷积编码/解码，以正常fit训练，MSE/Adam；固定10epochs、batch16、lr0.001。
按正常validation的逐图加权MSE保存最佳epoch，不使用test选择模型。训练加载器固定随机种子。
热图是RGB平均绝对重建误差。

图像分数均为热图最高1%像素均值。阈值仅用正常validation：image目标FPR5%，pixel1%；
统一严格 `score > threshold`。报告image AUROC/AP/FPR，pixel AUROC/AP及AUPRO。
AUPRO使用8邻域、200个阈值、FPR上限0.30，线性插值积分端点、除以0.30归一化；
曲线扫描不用于选择部署阈值。`03_evaluate.py` 的纯评估接口可供Part2复用。
模板配准/平滑与AE重建是方法内部步骤，均已记录，不能作为未注明的公共预处理。

## 从干净副本运行

在仓库根目录执行：

```powershell
python -m compileall -q part1
python part1/verify.py
python part1/01_template_baseline.py
python part1/02_autoencoder.py
python part1/03_evaluate.py
python scripts/verify_part1_release.py
```

单类快速运行（只验证流程，不作为正式结果）：

```powershell
python part1/01_template_baseline.py --categories candle
python part1/02_autoencoder.py --categories candle --epochs 1 --batch-size 8
python part1/03_evaluate.py --categories candle --methods template autoencoder
```

快速实验请指定另一个 `CVPROJECT_OUTPUT_ROOT`，避免覆盖正式10轮输出；正式发布核查器
会拒绝训练轮次/源码/样本与协议不符的输出。阈值参数和模型训练参数的具体选项见 `--help`。

## 结果、检查与限制

输出包含检查点、训练历史、validation/test逐图CSV和448×448 float32热图；evaluation保存
指标、完整FP/FN列表、每类最多6个错误面板及机器可读发布核查报告。
AE分别记录encoder、decoder/残差/传回CPU耗时、完整推理耗时、训练显存峰值与参数字节数；
模板记录建库和完整配准/差异评分耗时、模板字节数，无学习型特征提取阶段。
耗时包含的步骤不同，不能将单个字段直接称为同一口径的模型速度排名。

行为检查覆盖平移、旋转、边界和配准失败回退、低值mask、数据隔离、阈值并列分数、
完美/常数热图AUPRO。发布核查另逐张验证所有2568张预测热图、CSV分数和图像级指标，
并核对运行源码SHA-256。完整测试分数用于最终指标，不参与参数选择。

旧256结果、旧分支默认20轮AE及本地学习单模板结果不与本次448结果混用。
Part2/3和报告仍需继续；全项目至少20例FP/FN人工原因分类若未完成，会在结果状态中
明确标注，不能把自动错误清单当作原因分析已完成。

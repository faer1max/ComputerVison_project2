# Part 1：VisA 单类异常检测基线

Part 1 用 VisA 官方 1-class 划分实现两个可复现的无监督异常检测基线：基于图像模板的配准差分，以及只用正常训练图训练的卷积自编码器。统一评估脚本计算图像级和像素级指标，并导出错误案例。

## 数据与类别

将官方 VisA 数据解压到 `data/VisA/`，至少应包含 `split_csv/1cls.csv` 和 `candle/`、`capsules/`、`cashew/`、`chewinggum/`、`fryum/`、`pipe_fryum/` 六个类别目录。CSV 内的图像和掩码相对路径应能在该目录下找到。数据本身不随代码上传。

每类的官方训练划分只含正常图。脚本从该训练划分中固定随机划出 10% 作验证，训练不接触测试标签或掩码。测试集包含正常和异常图。

## 环境

在已配置好的 `cvproject` 环境中安装 `part1/requirements.txt`。PyTorch 请按本机 CUDA/CPU 情况单独安装；requirements 文件不锁定特定 CUDA 轮子。脚本在 CUDA 不可用时会退回 CPU。

## 运行

从项目根目录执行：

```powershell
conda activate cvproject

# 模板配准差分基线
python part1/01_template_baseline.py

# 正常图卷积自编码器（默认 20 epochs）
python part1/02_autoencoder.py

# 汇总两个方法的全部类别指标与错误案例
python part1/03_evaluate.py
```

可以先用一个类别做快速试运行：

```powershell
python part1/01_template_baseline.py --categories candle
python part1/02_autoencoder.py --categories candle --epochs 1 --batch-size 8
python part1/03_evaluate.py --categories candle --methods template autoencoder
```

自编码器支持 `--seed`、`--epochs`、`--batch-size`、`--learning-rate` 参数；两个推理/评估脚本均支持 `--categories`。模板基线默认每类最多取 24 张正常训练图构造模板。

## 评估约定

- 图像分数是异常热图最高 1% 像素的均值。
- 图像阈值在正常验证图上按目标 5% FPR 校准；像素阈值按正常验证像素的 99 分位数校准。测试标签仅用于最终计算指标。
- 报告图像 AUROC/AP、像素 AUROC/AP、AUPRO（积分范围 FPR 0–0.3）和校准阈值下的图像 FPR。
- VisA 部分原始掩码的前景值是 1 或 6，因此读取时以 `mask > 0` 转成二值图，而不是用 127 作阈值。

## 输出

所有运行产物写入被 Git 忽略的 `outputs/part1/`：各方法/类别目录包括验证与测试预测 CSV、NumPy 热图、训练历史和检查点；`evaluation/` 保存汇总指标、协议 JSON、错误列表及错误案例拼图。完整六类自编码器运行耗时取决于 GPU 和 epoch 数。

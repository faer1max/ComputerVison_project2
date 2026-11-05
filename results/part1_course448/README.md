# 2026-11-05 Part 1 真实重跑结果

本目录只含可分享摘要与代表图；数据、权重、完整预测及热图保存在运行目录，不随Git上传。
模型代码对应源码提交 `86e127e` 的Part1文件及共享协议，精确SHA256见
release_verification.json；后续结果提交未更改这些模型源码。

验证量：六类×两方法；324张正常validation和960张test各预测两次，**2568张448×448
有限值热图全部核对通过**。六个AE均实际训练10轮；训练历史、最佳epoch、检查点类别和
输入尺寸均核对。测试包含360正常/600异常，官方测试图不进入训练或阈值选择。

| 方法 | image AUROC | image AP | test FPR | pixel AUROC | pixel AP | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 24图均值模板+ECC差异 | .6562 | .7661 | .0828 | .8131 | .1624 | .5543 |
| 正常图AE（10epochs） | .7680 | .8408 | .0583 | .6941 | .0312 | .5580 |

以上为六类别简单宏平均，不是合并全部图像的全局指标。逐类结果见metrics.csv，完整
精度见macro_mean.csv。正常验证集目标image FPR=5%不能保证分布不同的test仍为5%。
没有根据这些test结果重选阈值、训练轮次或模板；结果低也保留。

旧256、单图模板及旧20轮上传代码与此次有多个设置差异，不能把新旧指标差异归因于
某一个修改。模板坐标修复由独立平移/旋转检查验证，不以指标提高与否证明正确性。

文件说明：

- run_metadata.json：12组方法/类别的运行身份、耗时、配准回退、AE训练资源与参数。
- evaluation_protocol.json、根experiment_protocol.json：输入、阈值与AUPRO协议。
- release_verification.json：完整样本/热图/源码/图像指标的核查报告。
- environment.json/environment.txt：实际版本和环境快照；本机RTX4060 Laptop，Python3.11。
- inference_efficiency.csv/efficiency_protocol.json：训练结束后串行推理测试，同一批每类16张
  正常validation图；AE3张预热排除，区分预处理、encoder、decoder/评分和纯推理GPU峰值。
  模板没有学习型encoder，空字段表示不适用；其ECC/差异耗时包含在评分里。
  AE训练峰值在run_metadata中，不与这里的推理峰值混用；时间不是其他硬件的速度保证。
- error_cases.csv/error_counts.csv：完整FP/FN清单与计数；只使用相对图像路径。
- visual_cases.csv、visual_cases_1–4.png、visual_case_statistics.csv、FAILURE_ANALYSIS.md：
  24张不同图像的逐例视觉复核、分类和数值证据；原因是待验证假设，作者最终复核待办。
- demo_template/autoencoder.png与.json：真实单图分数、OK/NG、热图及像素预测；000.JPG
  是异常样本，两基线均漏检，作为诚实的失败演示保留。运行脚本见Part1 README。

源码验收：干净Git归档中的compileall、三个CLI、7项行为检查，以及全量输出核查。
源码提交、工作目录和归档的算法文件字节一致。发布前还会对最终结果提交重复检查。
Part2/3、最终报告和demo打包不由本次Part1上传自动视为完成。

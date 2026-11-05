# Part 1 发布验收口径

修复计划先于本次实现确定：从原part1complete分支隔离修改；固定448协议；修复语法与
像素坐标；六类重新运行；检查精确提交；最后上传并核对远程。当前学习目录与旧结果保留。

| 要求/风险 | 实现与验证证据 |
| --- | --- |
| 官方六类及划分 | common读取官方1cls.csv；verify检查fit/validation/test互斥。 |
| 相同输入和掩码 | experiment_protocol.json固定RGB/bilinear448、nearest/>0；AE448/16=28，基础模型448/14=32。 |
| 只学正常数据 | AE训练Dataset只有正常fit，best epoch按正常validation；模板仅来自fit。 |
| 禁止test选阈值 | calibrate_threshold与calibrate_pixel_threshold只接正常validation；判定统一为严格>。 |
| 可用语法与命令 | 干净副本compileall、三个入口--help、7项行为检查；README命令与实际CLI一致。 |
| 定位坐标一致 | ECC矩阵方向明确；平移/旋转/无覆盖边界/配准失败测试，输出在原测试图坐标。 |
| 图像与像素指标 | 同一03_evaluate.py评估两方法；AUROC/AP/FPR、pixel AUROC/AP、AUPRO .3/200/8邻域。 |
| 训练/运行真实性 | 所有类别重新运行；每类元数据记录源码SHA256、输入尺寸、训练轮次；不沿用旧256结果。 |
| 输出完整与可追溯 | verify_part1_release核对2568张热图及预测CSV、重算图像分数和指标、核对源码字节。 |
| 效率 | AE encoder与decoder/评分时间、峰值GPU分配、参数/检查点大小；模板建库/评分时间与模板大小。 |
| 定性与demo | 导出失败列表和输入/GT/热图/阈值面板；part1_demo读取已校准阈值输出分数/OK-NG/热图。 |
| 发布文件安全 | 不提交数据、模型、依赖环境、绝对机器路径；移除个人教学上下文；只分享摘要与代表图。 |

机器验收是否通过以 `results/part1_course448/release_verification.json` 为准，完整来源见
同目录 `run_metadata.json`；`environment.json`和environment.txt记录本次实际环境。
即使指标较低也如实发布，不据test调参后替换为最好结果。

边界：本分支先交付Part1。Part2的三次随机参考1/2/4/8-shot、AnomalyCLIP及Part3探索，
全项目20例原因分析、最终报告和demo打包均不能从Part1代码上传推断已完成；各自状态需另行确认。

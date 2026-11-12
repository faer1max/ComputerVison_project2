# Part 1与Part 2：对照与失效分析

依据课程AIAA3201_Project2_Few_Shot_Industrial_Visual_Anomaly_Detection_Spring2027.pdf第3页5.1、第4页5.2和第5页6.2。

## 共同协议与比较预算

四种方法共用六类官方测试集和normal fit/validation拆分、RGB bilinear448几何输入、nearest/>0掩码及Part1评估器。448是本项目选择的共同分辨率，原PDF要求一致而没有规定必须448。模型权重需要的归一化属于显式模型接口，不能将DINO/CLIP归一化后的Tensor称为相同。

模板法使用正常fit前24张中配准成功的参考求均值，AE使用完整正常fit训练10epochs、正常validation选最佳轮。DINO使用正常fit随机1/2/4/8张、三个种子，正常参考旋转8次；CLIP使用MVTec训练的官方提示，不在VisA梯度优化。预算不同，不能把表格称为“相同正常训练样本数”的消融。全部操作阈值只从正常validation确定，不用test重选阈值或超参数。

统一结果表见../results/part12_comparison/macro_metrics.csv。common_heatmap_top1pct是所有方法在448热图最高1%像素的均值，适合共同聚合对照；official_native另列DINO块top1%与CLIP全局图文概率，不混成一条性能指标。先每个seed平均六类，再对DINO三个seed计算均值和样本SD。模板、AE和确定性CLIP没有三次参考抽样，未人为生成其SD。每个配置都保留，不能只挑最佳shot/seed。

## 纹理、结构与光照：两基线的差别

模板差异依赖像素对应和正常模板的代表性。ECC减少整体旋转/平移，但不能将多个胶囊的相对排列、食物的形状变化及背景纹理一一消除。正常图C01/C05/C07/C09/C11的轮廓响应构成误报现象；不能直接认定ECC实现错误，也不能把正常姿态变化当成真实缺陷。

AE的正常重建目标不保证异常语义检测。C13/C15/C21/C23显示正常边缘/反光的残差，C17/C18显示高频背景残差；C19结构缺损和C22细线缺陷未明显脱离普通边缘响应。与模板相比，AE可以学习部分正常外观，但当前紧凑结构及训练设置不能保证其重建残差集中在缺陷。12行定量指标不能代替对这些模式的解释。

局部外观/纹理异常与结构缺损的对照见Part1 FAILURE_ANALYSIS.md逐例表：C02/C04/C06/C12/C14/C16/C18/C22/C24属小斑点、细线或局部弱响应；C08/C19/C20包含缺损/损伤。类别是观测与候选机制，不是数据集官方defect-type标注，不能据此编造按真实缺陷类型分层的整体性能。

光照实验由scripts/illumination_course448.py生成。固定已保存448模板、最佳AE和原正常阈值，对同一324张正常validation图施加0.75/1.0/1.25乘法，再clip并uint8截断。两方法输入完全相同，未在扰动后重校准。逐图分数留在本地outputs，逐类及汇总表、核验记录位于results/part1_course448/illumination。1.0条件必须逐图复现原评分及OK/NG；CSV评分表示与旧校准相同，并保留重算raw分数。

变亮条件可能高光裁剪，实验只测合成亮度下正常图误报，不测缺陷召回，也不等同于真实照明、视角或跨域稳健性。使用本次448数字，不混入旧学习版256实验。汇总同时给六类宏平均和按324图合并的FPR，二者不能互换。

## 冻结语义特征为何能定位未见缺陷

DINOv2将局部输入映到预训练表示空间；参考图提供的是目标产品正常局部的分布。测试块远离正常参考块时距离变大，因此无需目标类别缺陷标签也可能定位异常。它并非已经学会本数据集所有缺陷，正常外观差异、参考覆盖不足也会产生较大距离。官方PCA前景筛选和参考增强属于方法的测试时处理，已记录，不能隐藏成公共输入处理。

CLIP将图像与正常/异常文本放到共同语义空间；全局概率回答整图相似性，局部patch相似度生成热图。语言提示不保证对每个微小材料缺陷敏感，全局图像分数与局部定位可以不一致。零样本指目标数据未训练骨干/提示；正常validation阈值校准另行记录。

## 粗patch失效的具体证据

ViT patch14在448输入上是32×32网格。单块覆盖14×14输入像素，多种外观在块特征中混合；448热图的插值和平滑不增加新的细节证据。因此小于块的斑点、细线可能有弱响应，邻近高响应也可能扩散出标注。不能把一个宽热区解释为精确恢复了细小缺陷边界，也不能据一个失败图证明是patch大小单一因素导致。

../results/part12_comparison/capsules_detail.png和chewinggum_detail.png按固定类别的第一张异常图展示整图/GT包围框局部、DINO seed42/1-shot与CLIP。裁剪只用于可视化，未送入模型或用于重新评估，色标沿用整图范围。它们比较微小缺陷与较大损伤的响应，提示未来更细特征/多尺度实验的动机，但本轮不把尚未进行的消融写为已证明改进。

## 失败分类复核与解释边界

2026-11-12再次目视复核Part1的四张6例总图，24张独立产品图均有输入、GT、热图及阈值预测。C01–C12显示模板轮廓/排列/背景响应与局部缺陷弱响应；C13–C24显示AE普通边缘、背景残差、细小外观或结构缺损弱响应。逐例分类仍见results/part1_course448/FAILURE_ANALYSIS.md，机器核对见visual_case_statistics.csv。此为助手视觉复核，不能冒充作者签名或受控因果证明。

图像FP/FN、像素召回和背景误报必须分别看：图像FN也可能局部定位正确但分数不过阈值；高图像AUROC也不保证当前正常FPR操作点的漏检低。Part2另列24个模型-图片案例，不声称24张独立产品图。读图后的现象解释没有回流为本轮测试阈值/参考/轮数的选择。

## 效率与引用

Part1和Part2资源记录已包含提取/评分、PyTorch分配峰值及参考库字节。模板无神经特征提取，CLIP无正常参考库，两者写不适用而非遗漏。DINO缓存、恢复累计计时重置及部分并行负载已有说明；现有时间只报告运行口径，不进行隔离端到端FPS排名。代码源码与权重身份见assets/source_revisions.json和part2/assets_verified.json；数据图及标注来源VisA/SPot-the-Difference，预训练模型来源DINOv2及CLIP，具体官方链接见PREPARATION.md与literature/sources.json。


## 本轮补充实验与核查命令

在仓库根目录、已准备数据与Part1模型后执行；若产物在别处，用CVPROJECT_DATA_ROOT/CVPROJECT_OUTPUT_ROOT指定原数据和Part1输出，CVPROJECT_ASSETS指定Part2本地依赖，CVPROJECT_PART2_OUTPUT指定原Part2输出。

```powershell
python scripts/illumination_course448.py
python scripts/build_part12_comparison.py
python scripts/verify_part12_supplement.py
python part1/verify.py
python part2/check_contract.py
```

光照脚本无训练、不重新选择阈值；新demo沿用原float32 CSV评分表示并另报raw分数。重新执行会覆盖同名补充实验文件，请保存此前结果后再有意重跑。

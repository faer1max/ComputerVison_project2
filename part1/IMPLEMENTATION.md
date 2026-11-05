# 修复代码的数据流

本项目的函数、变量名可以自定义；`cv2.findTransformECC/warpAffine`、
`torch.nn.Module.forward`、NumPy/pandas库调用需遵循其API接口。

- `register_to_template(image, template)`：两个H/W/3 RGB uint8数组。转灰度float32/255，
  ECC估计template→test矩阵W，用inverse-map把测试图采样到模板坐标。返回aligned图、
  2×3浮点矩阵W及bool成功标志；失败返回原图、恒等矩阵、False。build_template和
  make_difference_map调用它，前者只纳入成功配准的拟合图。
- `restore_map(aligned_map, warp, fallback)`：H/W单通道热图、W与同尺寸未配准差异。
  先对全1有效域做双向映射，排除反射/无对应区；热图用W正向投影回测试坐标，缺失区
  使用fallback。返回float32 H/W热图及0–1覆盖率。平移测试中原x170–179缺陷仍在原处，
  不再停留在模板x150–159。旋转检查核对缺陷质心，边界检查保证未知区域不被伪造成0。
- `make_difference_map(image, template)`：注册→绝对RGB差均值/255→5×5、sigma.8平滑，
  同时计算未注册差异作边界回退→restore_map。返回热图、配准状态、覆盖率给predict_rows。
  模板定位不再直接用模板坐标的热图与原图掩码比较。
- 模板`predict_rows(rows, template)`：rows为DataFrame，每行有image_path等字段。
  逐图加载为448×448×3→热图448×448→top1%分数，积累Python list，返回maps、scores、
  quality及秒数。quality中的配准状态/覆盖率进入预测CSV。label不参与打分。
- 模板`run_category(category, seed)`：类别字符串、默认42；读官方划分，复用10%验证划分，
  构建均值模板，预测validation/test，保存CSV/热图/元数据；无返回值。模板数和失败数如实记录。
- AE `predict_rows(model, rows, device)`：训练后模型、图片表和torch.device。图片转成
  [1,3,448,448]，依次是一张图、RGB通道、高、宽；encoder输出[1,256,28,28]，
  decoder重建[1,3,448,448]，RGB绝对残差均值形成[448,448]热图。返回maps/scores/
  总耗时/分阶段耗时字典。GPU阶段同步后计时，避免只记录异步启动时间。
- AE `run_category`：正常fit训练10轮，validation按实际batch样本数加权，按validation
  最低MSE拷贝并恢复最佳state_dict，随后预测两组。test表不进入optimizer更新或最佳轮次选择。
- `evaluate_aupro(test_predictions, category, max_fpr=.3, num_thresholds=200)`：输入DataFrame
  含mask_path/map_path；读取H/W掩码和热图，8邻域连通分量，收集各区域与背景分数，排序后
  用searchsorted统计每个阈值的覆盖/FPR（避免反复扫描全图）。整理相同FPR点包络，插值.3端点，
  积分并归一化；返回AUPRO、积分设置和区域数dict，供共享评估写入结果。
- `source_revision()`：无参数；读取本分支协议和Part1源码字节，返回相对路径→SHA256字典，
  每个类别元数据与上传前核查器用它核对源码身份。
- `verify_release()`：无参数；由common读取输出根目录。检查完整样本、形状/有限值、
  热图重算分数、源码哈希、图像AUROC/AP/FPR及正常验证阈值，返回报告dict。主入口写JSON。

本版本统一输出根目录为outputs/course448/part1，可用环境变量覆盖；数据根目录同样可配置。
模型、数据和完整热图均不随Git上传。正式运行输出与上传代码由哈希建立联系，而非仅靠文件名。

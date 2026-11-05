# Part 2 固定执行方案

沿用Part1 c679ca1的共享448协议和评估代码。DINO六类×shots1/2/4/8×seeds42/43/44共72组，CLIP六类零样本共6组；正常validation与参考图互斥。官方DINO ViT-S/14、8角度参考增强、官方PCA测试前景策略、CPU FAISS精确检索；CLIP ViT-L/14、MVTec提示权重、DPAM20/features24/sigma4。冻结全部模型，无VisA拟合。缓存输入特征，只复用同源码身份缓存。原生评分与统一热图评分分列，不能把二者当成同一结果。保存validation/test逐图CSV、448浮点热图、原生score、参考图、时间/显存/记忆库字节及指标。验收78组/16692张图，核对随机参考和完整样本。上传前验证干净Git归档，上传后核对远程；不提交数据/缓存/权重/全量热图。

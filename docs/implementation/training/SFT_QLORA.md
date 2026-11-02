# SFT 低显存训练模式（QLoRA）

阶段工作台的 SFT 方案提供两种服务端固定模式：标准 LoRA（默认、兼容既有配置）和可选的 QLoRA 4-bit NF4。QLoRA 配置固定为 `bitsandbytes`、4-bit、NF4、FP16 compute、double quant；前端只能选择模式，不能自行提交量化细节。其他训练阶段不接受该模式。

选择 QLoRA 后，本地只读预检会检查当前项目 Python 环境的 `bitsandbytes`、PyTorch 与 CUDA 可用性。预检不会安装依赖、联网下载或自动回退到 LoRA。缺少依赖时必须先在项目虚拟环境中自行准备，再重新预检；确认令牌绑定所选模式和有效配置摘要，改变模式或配置后需要重新预检。

QLoRA 会量化基座权重，有助于降低权重显存；它不会消除激活、梯度和优化器状态等开销，也不保证给定序列长度一定能跑完。序列越长越容易耗尽显存，最终仍需在目标 GPU 和目标长度做有界实测。训练 Run、resolved config 与 lineage 会记录实际模式和摘要；训练或适配器导出/合并失败的产物不能作为下游成功模型。

## 本地依赖验证

在项目虚拟环境中自行安装并核验与本机 PyTorch/CUDA 兼容的 `bitsandbytes` 版本。Launcher 不会代为安装。当前实现的自动兼容检查验证包可发现且 PyTorch CUDA 可用，但最终能否成功训练仍由有界 QLoRA 训练探针确认。

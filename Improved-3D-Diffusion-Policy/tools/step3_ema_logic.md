# 第 3 步：模型与 EMA，直觉说明及修改对照

## EMA 是哪一份模型

想象训练模型是正在练习的学生，每看一批演示，optimizer 就根据 loss 调整它的参数。
有些批次会把参数往这边推，有些批次往另一边推，最新一次更新难免带一点波动。

EMA 是这名学生的一份“平滑副本”，网络结构完全相同，也能独立做预测。
它没有自己的训练 loss，不做反向传播，也不由 optimizer 更新。
每次训练模型更新后，EMA 跟着接收新参数，通常只向新参数挪一小步：

    新 EMA 参数 = decay × 旧 EMA 参数 + (1-decay) × 当前训练参数

例如 decay=0.9、某个 EMA 参数为 1.0、当前训练参数为 1.2，更新后 EMA 是 1.02。
这样保存的是近期训练过程的平滑结果。它平均的是模型参数，不是动作输出，也不是演示数据。
当前代码有 warmup：开头 decay=0，直接追上训练模型，再逐渐增加平滑程度；公式本次不变。
原实现对 BatchNorm 的 weight/bias 和不需要梯度的参数直接复制，本次也保留这个例外。
平滑副本不保证每一次都比当前模型好，这仍需后续离线检查。

这里 EMA 包含整个 DiffusionImagePolicy：图像编码器、动作 U-Net、normalizer 等。
训练 loss 用 self.model，训练批次的采样检查优先用 self.ema_model；两份都随 checkpoint 保存。
它不改变 28 维关节顺序，仍为左臂 7、右臂 7、左手 7、右手 7。

## 为什么还要管 train / eval

PyTorch 的 train()/eval() 是“运行方式开关”，和“是否允许学习参数”是两回事。
eval() 不会自动关闭梯度：本项目 R3M 在 eval 模式下，卷积权重和 BatchNorm 的 weight/bias 仍可微调。
EMA 则额外使用 requires_grad_(False)，因为它只接收复制或平均后的参数。

BatchNorm 可以理解成图像特征的“标尺”：除了可学习的 weight/bias，还保存均值、方差和计数器。
这些统计量叫 buffer，不走 optimizer，但属于模型状态，影响预测并进入 state_dict。
本项目沿用 R3M 预训练时的这把标尺，固定其统计量；学的是特征参数。
如果不小心调用整个模型的 train()，它会递归打开 R3M 的 BatchNorm，标尺就会被当前小批次改变。

原训练循环在 epoch 末尾调用 policy.train()。启用 EMA 时，policy 指的是 EMA，
所以本来只应该用于采样的副本反而被切回训练模式；下一次采样可能改动它的 BatchNorm 统计量。
而原 EMA 更新只处理 parameter，不复制 buffer，EMA 的“标尺”可能与训练模型不一致。
这一步就是让两份模型的职责、参数和附属状态都明确，避免跨 epoch 后悄悄变样。

## 两个源文件具体怎么改

dp_workspace.py：

- 创建 EMA 副本后立即 eval 并关闭梯度，optimizer 仍只绑定训练模型。
- 新增 _set_model_modes(training)：训练模型整体按阶段切换，随后单独把 R3M 子模型恢复 eval；EMA 始终 eval。
- 每个 epoch 开头明确进入训练阶段，采样前两份都进入 eval；采样后只恢复训练模型的训练模式，R3M 和 EMA 保持 eval。
- 原 policy.eval()、policy.train() 留在原位置作为注释；新的模式处理及原因写在附近。
- 正常结束后同步写 latest.ckpt，避免最后几轮未到 checkpoint_every 时未保存。
  已有 BaseWorkspace 保存接口会包含配置、模型/EMA 权重和 buffer、各自 normalizer、optimizer 以及计数器。

ema_model.py：

- 原参数更新代码逐行保留，decay 公式及 BatchNorm 参数例外不变。
- 紧接参数更新增加 buffer 复制，包含浮点统计量和整数计数器；buffer 不再做第二次 EMA。
- 原来没有 buffer 同步代码可供注释，因此只添加缺失逻辑及说明，没有删除旧实现。

原验证开关仍为 false，test_mean_score 仍只是负训练 loss。
随机裁剪、旋转和颜色增强的实现未改：它们在 train/eval 中都会执行；eval 不等于关闭增强。
没有修改 3D workspace、梯度累积或续训机制。
这里的“结束保存”指正常跑完；中断恢复和 EMA 更新步数的断点续训精确恢复不属于本步。

## 如何核对

    /home/dingzirong/miniconda3/envs/idp3/bin/python tools/check_g1_dex3_ema.py

检查使用真实完整 Zarr 和正式网络，在 CPU 上跑 3 个 epoch、每轮 1 个 batch（batch_size=2），
仅用于验证模式切换和更新，不是正式训练。测试关闭日志联网，生产配置仍为 online。
每轮执行真实 loss、反向传播、optimizer、EMA 和完整 DDIM 采样；同时检查固定 BatchNorm、梯度、参数变化、
EMA 原公式及 buffer 一致性，并读取结束后的 checkpoint 核对最后状态。
还用一个小模型主动改变浮点/整数 buffer，确认新增同步能修复陈旧状态。

## 实际核对结果

2026-10-21，真实数据、正式网络、CPU 检查通过：

- 3 次 loss 均有限；每次反向有梯度的 208 个参数张量全部有限。
- R3M 卷积、BatchNorm 的可学习 weight、U-Net 输出层均有非零梯度，optimizer 更新后确实变化。
- 跨 3 个 epoch 检查模式，R3M 统计量始终逐项不变；EMA 始终 eval，无梯度，且不在 optimizer 中。
- EMA 前两步 decay=0，第 3 步为 0.4053964425，参数结果符合保留的原公式。
- BatchNorm、浮点和整数 buffer 同步通过；小模型主动改动源 buffer 后，EMA 仍与源一致。
- 每轮完整 16 步 DDIM 预测有限，返回 action 为 [2,15,28]，已知切片行为保留；eval 下随机增强仍执行。
- checkpoint_every=100 时，第 0 轮有周期保存，正常完成第 3 轮后另有结束保存。
  重新读取 latest.ckpt，配置、计数器、两份模型所有 state_dict 项（包括 normalizer 和 buffer）均与结束状态完全一致，optimizer 保存了 3 次更新的状态。
- 检查未运行验证、未上传日志。未做正式长训练，CPU 成功不代表 GPU 已可用。

结果：data/g1_dex3_ema_check/check_result.json。
检查用权重：data/g1_dex3_ema_check/checkpoints/latest.ckpt。
这是 3 个 batch 的链路检查权重，不作为正式训练结果；完整预测重载和过拟合检查仍按第 4 步实施。

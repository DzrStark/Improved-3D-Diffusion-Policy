# 第 2 步：接入 28 维训练配置

任务配置：diffusion_policy_3d/config/task/g1_dex3_pick_right-image.yaml。
主配置：diffusion_policy_3d/config/dp_g1_dex3_rgb.yaml。

## 28 维如何传入模型

第 1 步 Zarr 已按 left_arm[7]、right_arm[7]、left_ee[7]、right_ee[7] 拼好。
现有 GR1DexDatasetImage 直接读取 state/action，不按关节名重新拼接；
state 对应 obs.agent_pos，两个字段的最后一维都是 28，顺序保持不变。
shape_meta 声明 agent_pos=[28]、action=[28]，policy 根据 action.shape 构建 28 维 U-Net。
本步骤不更换 Dataset 类，不做关节或电机映射。

## 配置如何继承

主配置继承 dp_224x224_r3m，再通过 Hydra defaults 的 override task 替换其
不存在的 adroit_door_image 默认任务。只覆盖宇树任务、运行命名、日志与保存选项；
R3M 预训练和微调、随机增强、U-Net、DDIM、EMA、优化器和训练周期全部继承。
当前 goal.md 第 2 步要求 online 日志，因此 logging.mode=online。

val_ratio=0.0、max_train_episodes=null：全部 60 条训练，不抽取验证或测试集。
horizon=16、n_obs_steps=2、n_action_steps=16，沿用 pad_before=1、pad_after=15。
采样器在每条 episode 内建窗口，边缘重复该 episode 的首/尾帧。
Dataset 返回完整 16 帧 RGB、state、action；policy 取前 2 帧观测，监督完整 16 帧动作。
Dataset 的 RGB 为 [16,224,224,3]、float32、0–255；policy 内部除以 255 并转成通道在前。
action 保持原 limits 归一化，state 保持 identity。已知动作返回切片为 15 步，本步不改。

training.resume=false，日志也不续接旧运行，checkpoint.save_ckpt=true。
输出目录包含专用任务目录、日期、微秒时间及 seed；hydra.job.chdir=false，
从项目根目录运行时相对 Zarr 路径不会因 Hydra 切换工作目录而失效。
训练/EMA 模式与训练结束保存逻辑属于第 3 步，本步仅启用现有 checkpoint 机制。

## 检查与使用

先检查组合结果，不启动训练、不连接 WandB：

    /home/dingzirong/miniconda3/envs/idp3/bin/python train.py --config-name=dp_g1_dex3_rgb --cfg job --resolve

数据路径、GPU、seed 等仍可使用原 Hydra 命令行覆盖。
完成第 3–4 步的检查前，本步不启动正式训练。

复核全部数据、窗口、归一化和实际策略构建（不启动训练、不连接 WandB）：

    /home/dingzirong/miniconda3/envs/idp3/bin/python tools/check_g1_dex3_config.py

## 实际核对结果

2026-10-21 已通过：

- Hydra defaults 成功继承主配置并覆盖 task，未加载缺失的 adroit_door_image。
- policy、optimizer、EMA、dataloader 与原 R3M 配置一致；训练设置仅改变 resume。
- Dataset 读取全部 60 条、10,344 帧，训练 mask 全为 true，验证样本数为 0。
- 全部 10,404 个窗口的索引均位于单条 episode 内，padding 保持 1/15。
- 每条 episode 抽核首、中、尾窗口，共 180 个序列；图像、state、action 与 Zarr 对应行及边缘重复值逐项一致。
- 样本 RGB=[16,224,224,3]，state/action=[16,28]，均为有限 float32；保留原关节顺序。
- action normalizer 的 min/max 来自全部帧，state identity 核对通过。
- 实际加载本地 R3M 预训练权重并构建 DiffusionImagePolicy：动作、mask、U-Net 输入和输出均为 28 维，观测条件特征为 1080 维。
- online 日志、禁止续训、开启 checkpoint、专用输出目录均已核对；此次检查未创建 WandB 运行或启动训练。

实际组合配置：data/g1_dex3_config_check/resolved_config.yaml。
检查结果：data/g1_dex3_config_check/check_result.json。
本步只验证数据与模型配置接入；反向更新、EMA 状态和 checkpoint 保存重载按第 3–4 步继续。

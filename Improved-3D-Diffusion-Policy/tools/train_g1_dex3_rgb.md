# 开始训练 G1-D + Dex3 RGB 策略

## 正式训练命令

在项目根目录执行：

```bash
./tools/train_g1_dex3_rgb.sh
```

脚本显式使用 `dp_g1_dex3_rgb` 配置、数据 `data/dex3_pick_right_rgb.zarr` 和 GPU 0，
从任意工作目录以脚本路径调用都能定位仓库。每次默认创建新的日期/时间/seed 输出目录，已有输出目录会拒绝覆盖。
训练使用全部 60 条 episode，保存 latest 和周期 checkpoint；WandB 使用 online 模式，实时上传训练日志。

脚本依赖已安装项目依赖的 idp3 conda 环境。运行前激活环境并确认 GPU 可用：

```bash
conda activate idp3
wandb login
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

2026-10-21 的检查中，沙箱内 torch.cuda.is_available() 为 False，但批准在沙箱外执行后，
GPU 入口已完成 2 个真实 batch、离线日志和 checkpoint 保存。
正式长训练尚未启动；请在实际训练终端确认 GPU 可用后运行。

## 指定参数

```bash
./tools/train_g1_dex3_rgb.sh \
  --zarr data/dex3_pick_right_rgb.zarr \
  --gpu 0 \
  --seed 42 \
  --output-dir data/outputs/g1_dex3_rgb/run_seed42
```

支持 `--zarr PATH`、`--gpu INDEX`、`--seed INTEGER`、`--output-dir PATH`、`--python PATH`。
相对数据/输出路径以项目根目录为基准，`--help` 查看用法。
脚本通过 CUDA_VISIBLE_DEVICES 选择该 GPU，再使用可见设备 cuda:0。
默认解释器为 /home/dingzirong/miniconda3/envs/idp3/bin/python；也可用 --python 指定其他已安装依赖的解释器。

按最新要求，主 YAML 和训练入口均默认 logging.mode=online，日志上传到 WandB 的 g1_dex3_rgb 项目。
在训练终端先完成 wandb login，并保持网络可用；已登录的终端无需重复登录。
如临时需要离线日志，可在最后追加 -- logging.mode=offline。
训练仍保留随机图像增强、16 步监督及原有 15 步动作切片。

只查看配置、不开始训练：

```bash
./tools/train_g1_dex3_rgb.sh -- --cfg job --resolve
```

额外 Hydra 参数放在 -- 后，例如 -- dataloader.batch_size=16。
默认 301 个 epoch、batch_size=32、学习率 1e-4；除入口所列覆盖项外均继承原配置。
正常结束后保存 latest.ckpt；周期保存间隔仍为 100 个 epoch。
请使用新的输出目录；本入口默认 resume=false，不承诺从中断位置精确续训。

输出目录内的 .hydra/config.yaml 是实际生效配置，logs.json.txt 是训练指标，
wandb/ 是本地 WandB 日志，checkpoints/latest.ckpt 包含模型、EMA、normalizer 和 optimizer。
test_mean_score 只是负训练 loss；本任务没有独立验证集或测试集。

## 启动前离线检查

```bash
conda activate idp3
python tools/check_g1_dex3_training.py --device cpu
```

默认进行 1000 次拟合更新，CPU 下可能需要几分钟；每 25 次输出一次进度。
保持正式配置的学习率 1e-4、R3M/U-Net 和随机增强。
拟合检查用恒定学习率，直接调用 optimizer，不执行正式训练的 500 步 warmup 和 cosine 调度；
正式入口仍保留原调度设置。
检查真实 Zarr 的全部 60 条数据与采样边界；抽取第 1 条 episode 内两个不含 padding 的窗口，
用正式 R3M/U-Net 做完整 16 步反向更新、少量数据拟合和 DDIM 采样。
训练噪声、时刻和随机增强每步变化；只在测量前后及重载对比时固定种子，不关闭增强。
使用同一批训练数据比较前后 loss 和 action MSE，两者都下降超过 50% 才算拟合检查通过。
它说明这两条序列能够被拟合，不证明模型利用了每一种输入或能泛化到新场景。
这不创建验证集。

checkpoint 会加载到全新的 workspace，比较模型、EMA、normalizer、optimizer 及计数器，
两份策略在相同种子下分别核对完整预测与动作切片，允许最大绝对误差 1e-6。
检查结果、实际配置、拟合曲线与检查用 checkpoint 保存在 data/g1_dex3_training_check/<时间>/。
其中权重只供离线核对，不作为正式训练结果。

只做一个 batch 更新与保存重载、跳过过拟合：

```bash
python tools/check_g1_dex3_training.py --device cpu --overfit-steps 0
```

需要在 GPU 上运行离线检查时使用 --device cuda:0。--output-dir 可指定新的检查目录，
--overfit-steps 控制拟合更新次数；跳过拟合的报告会明确标记 overfit_performed=false。

需要通过入口跑一个 CPU 短程训练（不做正式训练）时，应先隐藏 GPU，避免 R3M 的 DataParallel 自动绑定 GPU：

```bash
./tools/train_g1_dex3_rgb.sh --gpu '' --output-dir data/outputs/g1_dex3_rgb/cpu_smoke \
  -- training.device=cpu training.num_epochs=2 training.max_train_steps=1 \
  dataloader.batch_size=2 dataloader.num_workers=0 dataloader.pin_memory=false \
  val_dataloader.num_workers=0 logging.mode=disabled checkpoint.topk.k=0
```

这里 logging.mode=disabled 适合禁止本地 socket 的沙箱；正式入口默认使用 online 并上传日志。

## 本步骤的编程逻辑

启动脚本只负责“定位项目 → 拼接 Hydra 参数 → 交给原 train.py”，不复制训练循环。
28 维直接沿用 Zarr 的 left_arm[7]、right_arm[7]、left_ee[7]、right_ee[7] 顺序。
离线检查复用已有 Dataset、compute_loss、optimizer、EMA、predict_action 和 checkpoint 接口，
只在检查脚本中构造两条序列的小批次，不修改正式数据或训练配置。
loss 前向 hook 核对 U-Net 实际收到和输出 [2,16,28]，确保完整 16 步参与监督。
单独核对 action = action_pred[:,1:16]，保留原 15 步切片。

先执行一次真实反向，检查全部梯度有限、R3M/BN 可学习参数/U-Net 确实变化；
随后用变化的 diffusion noise、随机 timestep 和图像增强反复训练小批次。
4 个固定种子的 loss 均值和固定种子的采样 MSE用于前后对照，模型及 EMA 的结果分别报告。
重载检查恢复全新的 workspace，而非继续使用原内存对象；分别核对模型、EMA 的所有参数、buffer、normalizer，
并核对 optimizer 与更新计数。检查过程完全不初始化 WandB。

本步未修改既有源文件或作者 GR1 启动脚本。检查权重、短程运行与正式训练输出目录互相独立。

## 实际检查结果

2026-10-21 已通过：

- 60 条、10,344 帧全部可用于训练，10,404 个窗口不跨 episode；无验证样本。
- 完整 [2,16,28] loss/反向通过，208 个有梯度的参数张量全部有限，R3M 与 U-Net 确有更新。
- 两条训练序列、1000 次更新后，4 个固定种子的平均噪声 loss 从 1.111927 降为 0.157373；
  固定种子 DDIM action MSE 从 0.281090 降为 0.123579 rad²，两项均下降超过 50%。
  EMA 对应为 loss=0.194129、action MSE=0.130369 rad²。
- BatchNorm 统计量固定、normalizer 保持一致；随机增强与 15 步动作切片均保留。
- 全新 workspace 重载后，两份模型全部 state_dict、normalizer、optimizer 完全一致；
  固定种子的完整预测和动作切片最大绝对差都为 0。
- 此前 GPU 入口在 offline 模式下完成 2 个 epoch、每轮 1 个 batch，产生本地 WandB 文件、训练日志和最终 checkpoint。随后按最新要求将入口默认改为 online，已核对最终组合配置；尚未运行 online 训练。
- 跨工作目录调用、含空格路径、GPU 选择、seed、解释器、额外 Hydra 参数及已有目录拒绝覆盖均已核对。

完整拟合报告：data/g1_dex3_training_check/2026.10.21-16.03.24.704375/check_result.json。
对应检查权重和实际配置在同目录 checkpoints/latest.ckpt、resolved_config.yaml。
GPU 入口短程检查目录：data/g1_dex3_entry_gpu_offline_check/。

这些结果仅说明离线训练链路正常和少量训练序列可以拟合，仍有动作误差，
不代表已完成正式训练、收敛验证或真机成功率评估。

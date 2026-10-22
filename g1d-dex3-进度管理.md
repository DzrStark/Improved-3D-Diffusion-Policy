# G1-D + 双 Dex3 项目进度管理

更新时间：2026-10-22  
项目根目录：当前文件所在的 Git 仓库根目录  
训练代码根目录：`Improved-3D-Diffusion-Policy/`

本文是后续 Agent 的主交接文档，记录已完成工作、可复现入口、已知限制和下一阶段顺序。继续工作前先读本文，再按需要阅读对应源文件和专题报告。不要把离线结果解释成真机验证。

## 当前状态

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 数据整理与转换 | 已完成 | 七组 JSON/JPEG 转 Zarr 并检查；转换工具已迁入训练代码 `tools/` |
| G1-D RGB 训练代码 | 已完成 | 28 维配置、EMA 状态和训练入口已实现 |
| 四个正式训练 | 已完成 | right_pick、right_place、stack、right_to_left_handover_low，末轮 epoch 300 |
| 离线部署与真实演示回放 | 已完成 | 正式 EMA、输入处理、采样边界和离线调度已核验 |
| 新机器依赖全新安装 | 未验证 | 环境配置基于现有 Python 3.8 环境整理，不是已证明可重建的 lockfile |
| 真机 DDS、相机与状态订阅 | 未完成 | 没有可直接启动的硬件 transport；未连接机器人或发布电机命令 |
| 硬件映射、安全边界和闭环 | 未完成 | 关节坐标/零位/增益/限位/力矩、急停和任务成功率均未实机确认 |

目前复现流程和迁移后的代码检查见 [REPRODUCING.md](REPRODUCING.md) 与 [deploy/REPRODUCTION_AUDIT.md](deploy/REPRODUCTION_AUDIT.md)。

## 数据

### 数据位置与规模

这两个数据根目录位于 Git 仓库根目录，已被 `.gitignore` 忽略，不会随代码 push：

- 原始 Unitree JSON/JPEG：`tsingtec_raw_data/`
- 转换后的 Zarr v2：`tsingtec_training_data/`

每个转换数据子目录本身就是 Zarr store，并包含转换报告。七组数据全部是 60 个 episode，共 420 个 episode、76,795 帧：

| 原始任务 | Zarr 目录 | Episodes | 帧数 | 正式训练 |
| --- | --- | ---: | ---: | --- |
| dex3_pick/left | `left_pick/` | 60 | 10,261 | 否 |
| dex3_pick/right | `right_pick/` | 60 | 10,344 | 是 |
| dex3_place/left | `left_place/` | 60 | 8,935 | 否 |
| dex3_place/right | `right_place/` | 60 | 8,357 | 是 |
| dex3_stack | `stack/` | 60 | 9,294 | 是 |
| dex3_handover/left_to_right_handover_high | `left_to_right_handover_high/` | 60 | 14,958 | 否 |
| dex3_handover/right_to_left_handover_low | `right_to_left_handover_low/` | 60 | 14,646 | 是 |

正式训练选了表中四个任务，各自使用全部 60 个 episode。未划分验证集或测试集；训练 loss 和 `test_mean_score` 不是独立测试指标。

### 数据契约

- 图像来自 `colors.color_1`，即单独保存的右目图像；OpenCV 解码后 BGR→RGB，完整视野用 `INTER_AREA` resize 至 224×224，不裁剪、不做 JPEG 再编码。
- Zarr 图像是 `uint8 [N,224,224,3]`、RGB、0–255；state/action 是 `float32 [N,28]`，角度单位 rad。
- 28 维固定为 `left_arm[7], right_arm[7], left_ee[7], right_ee[7]`。按各组原始 qpos 数组次序保留，左右手符号不改。
- state 是同一帧实测关节位置，action 是同一帧记录的绝对目标位置；转换不对齐、不插值、不重采样、不将 `±0.001` 清零。
- 每个 episode 有 `conversion_report.json`，含源 JSON SHA-256、帧边界、图像处理和关节顺序。
- right_pick 60 条 JSON 均没有关节名及 `hand_coordinates` 元数据。因此 28 维拼接顺序已核实，真实硬件关节名称、方向、符号和零位不能从数据证明。

转换入口：

```bash
# 在 Git 仓库根目录运行；转换器默认读取 tsingtec_raw_data 并输出 tsingtec_training_data
python Improved-3D-Diffusion-Policy/tools/convert_all_unitree_to_zarr.py

# 单独转换并完整核对 right_pick
python Improved-3D-Diffusion-Policy/tools/convert_unitree_to_zarr.py \
  --raw-dir tsingtec_raw_data/dex3_pick/right \
  --output tsingtec_training_data/right_pick
python Improved-3D-Diffusion-Policy/tools/check_unitree_zarr.py \
  --raw-dir tsingtec_raw_data/dex3_pick/right \
  --zarr tsingtec_training_data/right_pick --all-images
```

转换器拒绝覆盖已存在的 store。全量转换会处理七组数据；若只需要已经准备好的 Zarr，不要重复运行转换命令。

## 相对上游的代码改动

基础项目是上游 iDP3 仓库，G1-D 分支使用 R3M + 2D RGB Diffusion Policy。原 GR1/3D 主线保留；新增任务配置复用原 Dataset、policy、U-Net、DDIM 与 EMA 训练框架。

### 数据和训练

| 文件/模块 | 本项目改动 |
| --- | --- |
| `Improved-3D-Diffusion-Policy/tools/convert_unitree_to_zarr.py` | 新增 JSON/JPEG→Zarr 转换；显式 right-eye RGB 和 28 维关节顺序，保存 episode/source hash 与转换元数据 |
| `Improved-3D-Diffusion-Policy/tools/convert_all_unitree_to_zarr.py` | 新增七组数据批量转换和独立核验入口 |
| `Improved-3D-Diffusion-Policy/tools/check_unitree_zarr.py` | 新增独立检查，逐项核对 state/action、episode 边界和 RGB 像素；支持搬迁后按内容/hash 校验 |
| `diffusion_policy_3d/config/task/g1_dex3_pick_right-image.yaml` | 新增 224×224 RGB 与 28 维状态/动作任务配置；right_pick 全部数据训练，`val_ratio=0` |
| `diffusion_policy_3d/config/dp_g1_dex3_rgb.yaml` | 新增 R3M RGB 训练配置；继承上游模型和优化器，配置 WandB、checkpoint 和独立输出路径 |
| `diffusion_policy_3d/workspace/dp_workspace.py` | 修复训练/采样阶段 train/eval 切换；保持 R3M BN 为 eval，EMA 常驻 eval 且无梯度；正常训练结束同步保存最新 checkpoint |
| `diffusion_policy_3d/model/diffusion/ema_model.py` | 在保留原 EMA 参数更新公式的同时同步模型 buffer，包含 BN 统计量及整数计数器 |
| `tools/train_g1_dex3_rgb.sh` | 新增可指定 Zarr、GPU、seed、输出目录和 Python 的训练入口；相对路径按代码目录解析 |
| `tools/check_g1_dex3_config.py`、`tools/check_g1_dex3_ema.py`、`tools/check_g1_dex3_training.py` | 新增配置、EMA、真实数据训练和 checkpoint 保存/重载检查 |

`GR1DexDatasetImage`、`DiffusionImagePolicy`、R3M ResNet18、U-Net、DDIM 和 normalizer 均复用现有实现，没有另写一套模型。策略契约为 `horizon=16, n_obs_steps=2, n_action_steps=16`；原 policy 切片 `action_pred[:,1:16]` 返回 15×28，保留原行为。训练监督完整 16 步。

### 离线部署

`deploy/g1dp/` 新增策略加载、观测预处理、数据源、执行过滤、调度及适配器协议；`deploy/replay.py` 与 `deploy/verify.py` 提供离线回放和完整 right_pick 检查。部署加载器从可信 checkpoint 严格恢复 289 个 EMA 参数/buffer/normalizer 张量，不扫描目录猜 checkpoint，不依赖 R3M pretrained cache，也不做联网下载。

`Session`、`JointOnlyRobot`、`ImageClientCamera` 和力矩 provider 目前是可注入的接入接口，不是完整硬件驱动。没有 DDS publisher/transport；离线入口不连接相机、SDK 或机器人。所有回放目标只记录，不会发到设备。

### 当前迁移修正

最近一次未提交的本地改动还修正了转换工具迁移后的默认路径、训练/部署默认 Zarr 位置、`verify.py` 的转换器引用、训练入口绑定开发机 Python 的问题。`goal.md` 已取消忽略并更新为当前状态。新增的总复现说明和测试仍需在提交前 review。

## 正式训练结果

四个正式训练均使用全量 60 episode、301 epochs（日志 epoch 0–300）、batch size 32、seed 42。任务没有 validation/test split；`test_mean_score=-train_loss` 是训练 workspace 产生的负训练 loss，不是测试成功率。

| 任务 | GPU | Epoch 300 train loss | 训练日志 `train_action_mse_error` | Global step | Checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| right_pick | 4 | 0.001691529 | 0.000285502 | 98,125 | `Improved-3D-Diffusion-Policy/data/outputs/g1_dex3_rgb/2026.10.21-16.25.46_gpu4_seed42/checkpoints/latest.ckpt` |
| right_place | 4 | 0.001989303 | 0.000255794 | 79,463 | `Improved-3D-Diffusion-Policy/data/outputs/g1_dex3_rgb/2026.10.21-16.36.38_right_place_gpu4_seed42/checkpoints/latest.ckpt` |
| stack | 5 | 0.001801163 | 0.000754942 | 88,192 | `Improved-3D-Diffusion-Policy/data/outputs/g1_dex3_rgb/2026.10.21-16.36.38_stack_gpu5_seed42/checkpoints/latest.ckpt` |
| right_to_left_handover_low | 5 | 0.001723917 | 0.000282406 | 138,459 | `Improved-3D-Diffusion-Policy/data/outputs/g1_dex3_rgb/2026.10.21-16.36.38_right_to_left_handover_low_gpu5_seed42/checkpoints/latest.ckpt` |

日志 loss 从 epoch 0 到 epoch 300 明显下降，但这只是训练数据上的拟合证据。没有独立验证集，不能据此推断泛化或任务成功率。权重约 1.44 GB/个，位于被忽略的 `data/`，**不会随 Git 上传**。SHA-256、配置、逐任务结果见 [training_inventory.json](deploy/reports/training_inventory.json) 和各 replay `summary.json`。

离线策略验证中，right_pick 相同输入及完整随机状态下，部署加载器与原生策略的 16 步预测和 15 步切片最大差均为 0。数据对齐时 15 步 action MAE 为 0.00333944 rad；分任务 replay 指标见 [OFFLINE_REPORT.md](deploy/OFFLINE_REPORT.md)。这些比较仍是训练演示的离线分析，不是闭环评估。

## 真机部署进度

### 已完成的离线部分

- 严格恢复正式 EMA 和 checkpoint normalizer；验证有限值、shape、权重 schema 和拒绝错误权重。
- 核对 right_pick 全部 60 条、10,344 帧 state/action 与 Zarr 一致；180 张抽样 RGB 与原转换路径逐像素一致；10,404 个 sampler 窗口均不跨 episode。
- 核对两帧历史、首帧重复 padding、episode reset、右目颜色和输入尺寸。
- 在四任务正式 checkpoint 和真实演示数据上运行离线 replay，记录预测、目标队列、逐部位误差及推理耗时。
- 实现测试用的队列、过期目标丢弃、位置/速率过滤和异常关闭逻辑；9 项执行测试加 3 项数据迁移测试通过。
- 对照 G1-D URDF 候选关节名并记录限位差异。right_pick action 有 38,791 个手部维度超出该 URDF 参考限位；没有把 URDF 或训练动作范围冒充硬件限位，也没有自动裁剪数据。

### 未完成且不可跳过的事项

1. **准备目标机环境和产物：** 安装/验证匹配架构的 Python、PyTorch/torchvision、CUDA/驱动；准备正式 checkpoint、所需 Zarr/原始数据及模型源码。目标机无 GPU 时需实测 CPU 延迟。
2. **只读接入：** 核实具体 G1-D、Unitree SDK、DDS 网络和相机服务；实现独立 subscriber，采集各关节组单独的 monotonic 接收时间、相机帧接收时间和消息序号。不能实例化构造时会发布命令的旧控制器来读取状态。
3. **确定坐标和控制权：** 对照真实设备确认 28 维顺序、方向、零位、拇指坐标、单位和非策略关节归属；将数据候选映射与硬件确认值分开记录。
4. **确认物理边界：** 现场确认关节位置/速度限位、控制增益、前馈力矩来源及对应硬件版本。需确认动力学模型与姿态；不能假定全零力矩等价，也不能把现有测试过滤参数用于真机。
5. **实现 joint-only transport 和 watchdog：** 只能写明获准的策略关节，明确 paused/stopped/fault 行为、看门狗、断联、异常退出、保持/释放策略。先完成不创建 publisher 的 shadow 模式。
6. **无运动验收：** 用真实相机和反馈跑 shadow，记录时序/新鲜度、推理耗时、过期目标、过滤比例；注入断流、NaN、写入错误等故障验证停止路径。
7. **运动阶段单独审批：** 在现场操作员提供初始姿态、关节组、动作幅度、限速/增益/力矩来源、急停和支持方式，并明确授权后，先做短时、小幅单关节/单组测试，再逐步进入 right_pick。未授权前不可切控制模式或发布电机命令。

当前离线 CPU 四线程推理约 160–180 ms，而 30 Hz 周期约 33 ms；默认 E=4 的目标窗口在该 CPU 环境会过期，Session 会丢弃过期目标，不补发。要在目标主机实测后选设备、E 和新鲜度阈值，不能声称当前满足实时性。

## 环境准备

现有验证环境为 Linux x86_64、Python 3.8.20、PyTorch 2.1.0+cu121、torchvision 0.16.0+cu121，CUDA build 12.1；检查环境时 GPU 不可用。`pip check` 在现有环境通过。部署依赖清单见 [requirements-offline.txt](deploy/requirements-offline.txt)，训练补充依赖见 [requirements-g1-rgb.txt](Improved-3D-Diffusion-Policy/requirements-g1-rgb.txt)，环境说明见 [IDP3_ENVIRONMENT.md](deploy/IDP3_ENVIRONMENT.md)。

部署新环境示例：

```bash
conda env create -f deploy/environment-idp3.yml
conda activate idp3-deploy
python -m pip check
python -m unittest discover -s deploy/tests -v
python -m unittest discover -s Improved-3D-Diffusion-Policy/tools/tests -v
```

从头训练时还需训练 workspace 的 WandB/pandas/tqdm 和本地 R3M/训练源码：

```bash
python -m pip install -r Improved-3D-Diffusion-Policy/requirements-g1-rgb.txt
python -m pip install --no-deps -e ./third_party/r3m
python -m pip install --no-deps -e ./Improved-3D-Diffusion-Policy
```

训练初始化使用上游 R3M pretrained loader；缺少 `~/.r3m/r3m_18/{config.yaml,model.pt}` 时可能尝试联网下载。部署加载正式 EMA 不需要此缓存；`deploy.verify` 的原生策略对照需要已有缓存。硬件接入另外需要经目标设备确认的 Unitree SDK/CycloneDDS、teleimager、Pinocchio 和匹配模型，当前没有锁定或验证这些硬件依赖版本。

部署 Python 环境配置只以现有环境为基线，**未证明从零安装可成功，也未验证 ARM/Jetson 或目标机 CUDA**。不要把完整 `environment-freeze.txt` 当成跨机器 lockfile，它含原机 editable 和 `/tmp` 路径。

## Git 与产物边界

当前数据、模型权重和训练输出都在 `.gitignore` 的 `data/`、`tsingtec_raw_data/`、`tsingtec_training_data/` 下，不进入 Git。复现者需要单独取得它们；目前仓库没有已知的 G1-D 数据或四个正式权重下载地址。README 中上游 iDP3/GR1 的公开数据链接不是本项目 G1-D 产物。

本总览及最近的迁移修正曾作为本地未提交改动生成。后续 Agent 应先运行 `git status --short`、审核 diff，并确认用户要保留/提交的范围；不要假设本文件或相关代码已经推到 GitHub。

## 主要文件索引

- 主训练目标和现状：本文件及 [Improved-3D-Diffusion-Policy/goal.md](Improved-3D-Diffusion-Policy/goal.md)
- 从 Git clone 到回放/训练的操作步骤：[REPRODUCING.md](REPRODUCING.md)
- 模型输入/输出及离线 replay 契约：[deploy/README.md](deploy/README.md)
- 四任务训练 loss、回放数值和限位比较：[deploy/OFFLINE_REPORT.md](deploy/OFFLINE_REPORT.md)
- 环境依赖和目标机迁移：[deploy/IDP3_ENVIRONMENT.md](deploy/IDP3_ENVIRONMENT.md)、[deploy/MIGRATION.md](deploy/MIGRATION.md)
- 源码证据和硬件控制风险：[deploy/SOURCE_AUDIT.md](deploy/SOURCE_AUDIT.md)
- 下一阶段 Agent 工作细化：[deploy/next_goal.md](deploy/next_goal.md)
- 发布前代码迁移复现检查：[deploy/REPRODUCTION_AUDIT.md](deploy/REPRODUCTION_AUDIT.md)

# G1-D + Dex3：单目 RGB Diffusion Policy 训练

## 当前状态与复现路径（2026-10-22）

四任务 `right_pick`、`right_place`、`stack`、`right_to_left_handover_low` 的正式训练均已完成，
各日志末轮为 epoch=300，正常结束的 latest checkpoint 计数为 epoch=301。
四任务正式 EMA 的离线推理已通过；没有真机闭环验证结果。
从全新克隆开始请先阅读 [复现说明](../REPRODUCING.md)，数据和正式权重需另行获取。

本文件位于内层训练代码根目录；下文 `tools/`、`data/` 均相对此目录，
`../tsingtec_raw_data/`、`../tsingtec_training_data/` 均从内层目录向上定位外层仓库中的数据目录。
转换、批量转换与复核脚本已迁入受 Git 跟踪的 `tools/`，不再使用原始数据目录下的脚本。
训练和部署默认均使用 `../tsingtec_training_data/right_pick`；旧 `data/dex3_pick_right_rgb.zarr`
仅为首轮转换产物，可显式指定。7 个短程检查 checkpoint 已于 2026-10-22 清理，
保留本机检查报告和日志；下文提到它们的保存/重载结果属于历史检查记录，不表示文件仍存在。

外部 URDF 原路径已变更；当前开发机位置为 `../dynamimic-deploy/reference/assets/urdf/g1_d_description/g1_d.urdf`（此路径相对外层 Git 仓库根目录），内容哈希与已有部署报告一致。数据、权重和 URDF 均不随 Git 上传。

## 目标与范围

- 基于 iDP3（arXiv:2410.10803v3）仓库的 **R3M + Diffusion Policy 2D 主线**，训练宇树 G1-D + 双 Dex3。
- 输入：右眼单目 RGB + 28 维实际关节位置；输出：28 维绝对目标关节位置，单位均为 rad。
- 28 维固定顺序：`left_arm[7], right_arm[7], left_ee[7], right_ee[7]`。保留组内原始顺序、左右手符号。
- 原始任务使用 `dex3_pick/right` 的 **全部 60 条 episode**；后来扩展的三个正式任务也各用对应全部 60 条，不划分验证集或测试集。右手任务仍保留完整双臂、双手 28 维。
- 当前已完成数据转换、四任务训练、checkpoint 保存及 `deploy/` 离线部署检查。实际通信、硬件映射和真机闭环仍待确认，见 `../deploy/next_goal.md`。
- 2026-10-21：第 1–4 步及第 5 步启动记录为历史过程；四任务训练已完成，当前状态以上方说明为准。
- 2026-10-21 16:39：按最新要求，另启动 `right_place`（GPU 4）、`stack`（GPU 5）、`right_to_left_handover_low`（GPU 5）三个独立任务，复用现有 RGB 训练入口；各用对应数据集全部 60 条、301 epochs、batch size 32、seed 42、WandB online。原 `right_pick` 继续运行；未使用 GPU 6。

## 数据与转换约定

原始数据根目录：`../tsingtec_raw_data/dex3_pick/right`。

原开发机上的参考说明（随数据单独获取，不在 Git 仓库中）：

- 原始数据：`../tsingtec_raw_data/dex3_pick/README.md`。
- 作者数据：`../yanjie_data/README.md`。

直接 **JSON + JPEG → Zarr**，不经过 HDF5。按 episode 编号、帧顺序读取：

| Zarr 字段 | 来源与格式 |
| --- | --- |
| `data/img` | 每帧 `colors.color_1` 指向的 `*_color_1.jpg`；RGB，完整视野缩放到 224×224；`uint8 [N,224,224,3]` |
| `data/state` | `states` 的四组 `qpos`；`float32 [N,28]` |
| `data/action` | `actions` 的四组 `qpos`；`float32 [N,28]` |
| `meta/episode_ends` | 各 episode 累计结束索引；`int64 [60]` |

- 约 30 Hz 原始帧序列原样保留；不做时间平移、插值、重采样或 action ±0.001 清零。
- 不生成 depth、point cloud 或 eef 字段；图像存 0–255，由现有策略除以 255，自带 R3M 内部已执行 ImageNet 标准化，避免重复处理。
- 默认输出 `../tsingtec_training_data/right_pick`；记录 episode 编号、帧数、关节拼接顺序和图像处理方式，便于追溯。
- 已扫描右侧 JSON 与图像路径：60 条、10,344 帧，关节维度正确、数值有限、右眼路径齐全；图像仅抽样解码，转换时仍需逐张读取。
- 已知时间间隔：`episode_0004/idx=123` 约 110 ms，`episode_0019/idx=114` 约 127 ms。保留数据，仅在转换报告中记录。

## 保留的原实现与核对结论

- 保留 R3M ResNet18 预训练与微调（`pretrained=True, frozen=False`）、U-Net、DDIM 和 EMA 主线。
- **按用户要求保留训练及评估/推理中的随机图像增强**，不增加关闭增强的改动。
- 保留 `horizon=16, n_obs_steps=2, n_action_steps=16` 及原动作切片逻辑；实际返回 15 步属于已知行为，本阶段不修正。训练仍监督完整 16 步。
- 保留 action 的 limits 归一化和 state 的 identity 处理；全部数据均用于训练，无验证统计泄漏问题。
- `dp_224x224_r3m.yaml` 默认引用不存在的 `adroit_door_image`，但作者训练脚本通过 `task=gr1_dex-image` 覆盖它；demo 不使用开门任务。新增入口明确选择宇树任务即可。
- 当前 `GR1DexDatasetImage` 读数逻辑不固定维度，可复用；不因类名含 GR1 而复制一个 Dataset。维度由新任务配置与实际 Zarr 共同确定。
- `DPWorkspace` 已关闭验证；保持不验证。`test_mean_score=-train_loss` 只是现有选模分数，不代表测试集或任务成功率。
- 已完成完整转换、28 维配置接入和四任务正式训练；第 3 步的 CPU 短程检查记录见下文。训练集检查不代表独立泛化验证。

## 按顺序实施

以下路径均相对于本文件所在目录。每步检查通过后再进入下一步。

### 1. 建立可追溯的训练数据

**全任务数据扩展已完成（2026-10-21，按最新要求）**：在工作区根目录新建
`../tsingtec_training_data/`，重新从原始 JSON/JPEG 转换 pick，并转换 place、stack、handover 全部数据。
七个直接子目录本身即 Zarr v2 store：`left_pick`（10,261 帧）、`right_pick`（10,344 帧）、
`left_place`（8,935 帧）、`right_place`（8,357 帧）、`stack`（9,294 帧）、
`left_to_right_handover_high`（14,958 帧）、`right_to_left_handover_low`（14,646 帧）。
每组 60 条，共 420 条 episode、76,795 帧；全部统一使用右目 `colors.color_1`，
沿用 224×224 RGB 与固定 28 维 state/action 约定，全部原始帧保留。
独立复核全部 state/action、全部 76,795 张 RGB 的逐像素一致性、源 JSON 哈希、episode 边界及图像块读取，均通过；
现有 ReplayBuffer 成功打开七组数据。各组保存转换和复核报告，总表见
`../tsingtec_training_data/conversion_summary.json`，读取检查见 `replay_buffer_check.json`。
批量入口为 `tools/convert_all_unitree_to_zarr.py`，复用 `tools/convert_unitree_to_zarr.py`；
`check_unitree_zarr.py` 已支持任意组的实际帧数及 `--all-images` 全图复核。
本次完成数据准备，以下第 2–5 步仍描述原 right pick 的配置和训练流程。

**首轮转换记录（2026-10-21）**：旧输出为 `data/dex3_pick_right_rgb.zarr`，共 60 条、10,344 帧。全部 state/action 与源 JSON 转 float32 后逐项一致；首、中、尾 episode 的 9 张 RGB 抽样逐像素一致，全部图像块可读，现有 ReplayBuffer 可直接打开。当前编程逻辑见 `tools/step1_data_logic.md`，转换/复核脚本均在 `tools/`，转换报告在对应输出目录的 `conversion_report.json`。当前统一数据位置是 `../tsingtec_training_data/right_pick`。

`tools/convert_unitree_to_zarr.py`：

- 提供原始目录、输出路径参数；仅使用 `color_1`，按固定 28 维顺序展平 state/action。
- OpenCV 解码时明确 BGR→RGB，统一缩放；按 episode 写入 Zarr，使用分块压缩，避免一次堆积全部原尺寸图像。
- 输出转换摘要与元信息；只检查必要的文件可读性、维度、有限值和 episode 边界。
- 检查：完整转换后为 60 条、10,344 帧，三个 data 数组长度一致，`episode_ends[-1]=10344`；抽核首、中、尾 episode 的原始值与转换值、RGB 顺序及图像处理一致。

### 2. 接入 28 维训练配置

**已完成（2026-10-21）**：新增下述两个配置并通过 Hydra 组合检查。现有 Dataset 读取全部 60 条、10,344 帧，验证样本为 0；全部 10,404 个采样窗口均不跨 episode，抽核 180 个序列的形状、数据顺序和 padding 正确。实际加载本地 R3M 预训练权重，policy/U-Net 动作维数为 28。按本节当前约定使用 online 日志，resume=false、checkpoint 开启。编程逻辑见 tools/step2_config_logic.md，复核脚本见 tools/check_g1_dex3_config.py；实际配置与检查结果保存在 data/g1_dex3_config_check/。未启动训练。

新增 `diffusion_policy_3d/config/task/g1_dex3_pick_right-image.yaml`：

- 图像 `[3,224,224]`、state `[28]`、action `[28]`；引用现有图像 Dataset 和新 Zarr。
- `val_ratio=0.0`、`max_train_episodes=null`，确保 60 条均进入训练；保留原采样窗口及 padding 规则。

新增 `diffusion_policy_3d/config/dp_g1_dex3_rgb.yaml`：

- 继承现有 R3M + DP 配置，显式覆盖为上述任务；保持增强、网络、归一化和时域设置。
- 使用明确的任务日志名，使用online日志，开启 checkpoint 保存；首轮使用独立输出目录、`resume=False`，避免误载 GR1 权重。
- 检查：Hydra 成功组合，Dataset 读取全部 60 条；样本形状正确，采样窗口不跨 episode；policy 动作维数为 28。

### 3. 理顺训练模型与 EMA 的状态

**已完成（2026-10-21）**：下述两个源文件已修改，替换的旧模式切换语句保留为注释，新实现紧随其下。训练模型按阶段切换，R3M 保持 eval 以固定 BatchNorm 统计量、参数仍可微调；EMA 始终 eval 且关闭梯度。EMA 原参数公式及其例外不变，新增 buffer 同步。正常结束后同步保存 latest checkpoint。CPU 使用完整 Zarr、正式网络运行 3 个 epoch、每轮 1 个真实 batch（batch_size=2），loss/梯度有限、R3M 和 U-Net 参数更新、跨 epoch 模式、EMA 参数/buffer、eval 随机增强及最终 checkpoint 内容核对均通过。未运行验证或正式训练。直觉说明及修改对照见 tools/step3_ema_logic.md，检查脚本见 tools/check_g1_dex3_ema.py，报告与检查用 checkpoint 位于 data/g1_dex3_ema_check/。

修改 `diffusion_policy_3d/workspace/dp_workspace.py`：

- 明确优化模型与 EMA 模型的训练/评估模式；保留 R3M 当前固定 BatchNorm 统计量、参数可微调的行为，避免 epoch 切换意外改变它。
- 保持不运行验证，不引入验证选模；确保训练结束保存最新 checkpoint，保留配置、normalizer、模型及 EMA 权重。

修改 `diffusion_policy_3d/model/diffusion/ema_model.py`：

- 处理模型 buffer 同步，使 EMA 的 BatchNorm 等状态与约定一致；不改变现有 EMA 参数更新公式。
- 检查：一个真实 batch 的 loss、梯度均有限；optimizer 更新后可训练参数确有变化；模型模式符合约定，EMA 参数和 buffer 行为正确。
- 不顺带重构断点续训、梯度累积或 3D 分支；当前梯度累积保持 1。

### 4. 提供训练入口和离线检查

**已完成（2026-10-21）**：新增下述两个脚本，训练命令、编程逻辑与结果见 tools/train_g1_dex3_rgb.md。入口按最新要求默认 online 日志、全部 60 条训练、独立输出及保存 checkpoint。检查使用真实完整 Zarr，完整 16 步 loss/反向与 15 步动作切片通过；两条训练序列做 1000 次新噪声/随机增强更新，固定测量噪声 loss 从 1.111927 降到 0.157373，action MSE 从 0.281090 降到 0.123579 rad²。全新 workspace 重载模型、EMA、normalizer、optimizer 后状态一致，固定种子两份策略的预测差异均为 0。报告与检查权重位于 data/g1_dex3_training_check/2026.10.21-16.03.24.704375/。此前经批准在沙箱外以 offline 模式运行 GPU 入口，2 个真实 batch、离线日志及结束 checkpoint 通过，结果在 data/g1_dex3_entry_gpu_offline_check/。未启动正式长训练，少量数据拟合不等于泛化或部署验证。

新增 `tools/train_g1_dex3_rgb.sh`：

- 从脚本位置定位项目，显式选择新配置；支持 Zarr 路径、GPU、seed 和输出目录。
- 不改作者 GR1 启动脚本；默认全数据训练、保存权重、online 日志。

新增 `tools/check_g1_dex3_training.py`：

- 用真实转换数据检查 Dataset、完整 16 步 loss、反向更新及采样输出，保留已知 15 步切片行为。
- 提供小数据过拟合检查，用于确认 RGB/state→action 训练链路有效；它不改变正式训练使用全部 60 条的要求，也不构成验证集。
- 检查 checkpoint 保存、重新加载及 normalizer 一致性；固定种子后核对重载预测。检查结果只说明离线链路是否正常。

### 5. 正式训练与交付

**已完成（2026-10-22 核对）**：right_pick 使用 GPU 4、全部 60 条、batch size 32、301 epochs、seed 42、WandB online。输出 `data/outputs/g1_dex3_rgb/2026.10.21-16.25.46_gpu4_seed42`；末轮日志为 epoch300，正式 latest 和 epoch300 权重已生成。正式 EMA 恢复和原策略一致性结果见 `../deploy/OFFLINE_REPORT.md`。

**新增三任务也已完成（2026-10-22 核对）**：数据直接读取 `../tsingtec_training_data/<任务名>`，各全部 60 条。`right_place` 使用 GPU 4，输出 `data/outputs/g1_dex3_rgb/2026.10.21-16.36.38_right_place_gpu4_seed42`；`stack` 使用 GPU 5，输出 `data/outputs/g1_dex3_rgb/2026.10.21-16.36.38_stack_gpu5_seed42`；`right_to_left_handover_low` 使用 GPU 5，输出 `data/outputs/g1_dex3_rgb/2026.10.21-16.36.38_right_to_left_handover_low_gpu5_seed42`。各日志末轮均为 epoch300，均有正式 latest/epoch300 权重及已提交的离线回放报告。原机启动记录位于 `data/training_runs/2026.10.21-16.36.38_three_tasks/`，不随 Git 上传；可迁移命令见 `tools/train_g1_dex3_rgb.md`。

- 先通过完整数据转换检查、单 batch 反向、短程训练和权重重载检查，再启动全部 60 条的正式训练。
- 正式长训练前确认实际 GPU 可用；此前沙箱内检测 `torch.cuda.is_available()=False`，第 4 步在获准的沙箱外已用 `cuda:0` 完成 2 个真实 batch、离线日志和 checkpoint 保存。正式运行仍需在目标终端确认设备，不能仅凭 CPU 前向判断 GPU 状态。
- 交付 Zarr 路径与转换摘要、可复现启动命令、实际生效配置、训练日志和可加载 checkpoint。
- 观察 train loss、采样 action 误差及必要的分部位误差；无独立验证集，不声称已验证泛化能力或真机成功率。

## 完成标准

各任务全部 60 条演示、仅右眼 RGB、28 维反馈与 28 维目标角正确进入训练；训练可正常更新并产出可重载权重。增强与时域行为符合用户要求，原 GR1/3D 路线不被无关改动影响。当前完成正式训练及离线部署，真机联调保持后续独立阶段。

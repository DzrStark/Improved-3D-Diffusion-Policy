# G1-D + 双 Dex3：离线部署准备 Coding Agent 提示词

以下正文可直接复制到新的 Coding Agent 对话中。

---

请在当前工作区完成 G1-D + 双 Dex3 的 RGB Diffusion Policy 离线部署准备，实际编写代码、运行检查并交付文档。先完成所有无需连接真机的工作，把必须连接真机的事项明确留到最后。不要只给方案，也不要在完成离线工作前反复询问是否继续。本阶段不连接机器人、不发布电机命令、不启动已有遥操或真机执行程序。

## 一、项目与资料位置

项目根目录：

`/home/dingzirong/workspace/Improved-3D-Diffusion-Policy`

训练代码目录：

`/home/dingzirong/workspace/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy`

训练目标、约定和修改记录：训练代码目录中的 `goal.md`。同目录 `tools/` 下还有配置、EMA、训练入口和离线检查的 Markdown 说明，请结合实际代码阅读。

正式训练输出：训练代码目录中的 `data/outputs/g1_dex3_rgb/`。

其中 right_pick 的 checkpoint 相对于该输出目录为：

`2026.10.21-16.25.46_gpu4_seed42/checkpoints/latest.ckpt`

另外三个任务目录：

- `2026.10.21-16.36.38_right_place_gpu4_seed42`
- `2026.10.21-16.36.38_stack_gpu5_seed42`
- `2026.10.21-16.36.38_right_to_left_handover_low_gpu5_seed42`

这些目录已有 `latest.ckpt`、epoch 300 checkpoint、实际 Hydra 配置、日志和 loss 诊断。请核实实际产物，不把 `goal.md` 里的历史“训练进行中”当作当前状态。优先用 right_pick 跑通完整离线链路，其余任务复用同一入口，通过显式配置选择权重和数据。

原始数据：

`/home/dingzirong/workspace/Improved-3D-Diffusion-Policy/tsingtec_raw_data`

转换后的七组数据：

`/home/dingzirong/workspace/Improved-3D-Diffusion-Policy/tsingtec_training_data`

数采与硬件接口参考，实际目录名是 `data_collection`：

`/home/dingzirong/workspace/data_collection`

另一套部署准备代码：

`/home/dingzirong/workspace/dynamimic-deploy/DEPLOYMENT.md`

机器人描述：

`/home/dingzirong/workspace/g1_d_description/g1_d.urdf`

部署代码、文档和离线报告优先组织在项目根目录的 `deploy/` 中；按职责安排子目录，不覆盖本提示词。外部数采、参考部署、URDF 目录优先只读使用，不直接修改它们。

## 二、当前模型的明确契约

这是 iDP3 仓库中的 **R3M + 2D Diffusion Policy**，不是 PPO，也不是数采目录里的 StarVLA/Qwen 模型。

### 输入

- 最近两帧右目 RGB。
- 两帧对应的实际关节位置，每帧 28 维，单位 rad。
- 固定顺序：`left_arm[7], right_arm[7], left_ee[7], right_ee[7]`。
- 明确历史按时间从旧到新排列；不能用上一帧策略目标代替实际关节反馈。

### 图像

- 原始 `colors.color_1`，即头部双目图像的右半幅。
- 完整视野，不做工作区裁切。
- 原转换流程：OpenCV 解码、BGR→RGB、`INTER_AREA` 缩放至 224×224。
- 部署输入保持 float32 的 0–255，策略内部已有除以 255；不要重复归一化。
- R3M 内部已有相关标准化，不重复添加。
- 用户此前明确要求保留推理时随机增强，本阶段保持，不擅自关闭。
- Zarr 中的图像已是 RGB、224×224，读取它时不要再做 BGR→RGB；区分原始相机输入与已处理数据。

### 输出与时域

- 28 维绝对目标关节位置，rad，不是增量、速度、力矩或末端位姿。
- `horizon=16`，`n_obs_steps=2`，`n_action_steps=16`。
- 当前实际 action 切片为 `action_pred[:,1:16]`，返回 15 步，这是已知且明确保留的行为，不修改训练主干去“修复”。
- state 使用 identity normalizer，action 使用训练保存的 limits normalizer。
- 恢复正式 checkpoint 的 EMA 和 normalizer，不重新估计统计量。
- 原始数据名义约 30 Hz；不能套用 PPO 参考项目的 60 Hz。
- 即使是右手任务，仍保留完整双臂双手 28 维，不擅自丢掉左侧维度。

这些约定需要与实际配置、Dataset、policy 源码核对。

## 三、哪些参考可用，以及不能混用的地方

### 1. 优先检查数采入口

`/home/dingzirong/workspace/data_collection/run/scripts/teleop_dex3_capture.py`

已看到它使用：

- `teleimager.image_client.ImageClient`
- `G1_29_ArmController`
- `Dex3_1_Controller`
- `get_current_dual_arm_q()`
- `ctrl_dual_arm(target, tau)`
- 头部双目右半幅写为 `color_1`
- 实测角度写入 `states`，控制目标写入 `actions`。

继续追踪它实际使用的控制器与录制模块，确认关节组内顺序、左右手符号、图像来源和坐标约定。原始训练数据的 `joint_names` 为空，不能只凭数组长度推断电机映射。

### 2. 还可参考

以下路径均相对于 `/home/dingzirong/workspace/data_collection/`：

- `run/scripts/evaluate_place_live.py`
- `run/scripts/dex3_live_support.py`
- `docs/README_dex3_joint28_live_inference_zh.md`
- `docs/README_dex3_zero_offset_zh.md`
- `third_party/xr_teleoperate/teleop/robot_control/`
- `third_party/xr_teleoperate/teleop/teleimager/src/teleimager/image_client.py`
- `third_party/unitree_sdk2_python/`

注意已有执行器可能在初始化时就创建发布线程、发送保持命令或切换控制模式；不得通过实例化它们来做离线检查。

### 3. 数采目录包含多个历史模型和补偿版本

- 单帧输入、16 步输出、语言指令、StarVLA 服务都不能直接套到当前两帧 Diffusion Policy。
- 裁切、PIL Bilinear、JPEG95 等处理不属于当前 iDP3 转换契约。
- 文档出现过 `thumb_0` 的直接透传、±30°偏置、缩放加偏置等不同方案，且文档与现有代码可能不一致。
- 必须追溯本次训练数据的实际坐标和元信息；不要自动给旧数据加补偿，也不要根据最新遥操代码推断旧数据坐标。
- 数采使用 G1_29，而外部描述是 G1-D；应核对关节、限位和控制接口是否匹配，不能仅凭名称断言一致。
- 无法离线确定的硬件版本、零位和映射应写入待确认项，不编造值。按实证写好映射与配置，明确哪些仍待硬件确认。

### 4. dynamimic-deploy 的适用范围

这是另一套 102 维观测、10 维动作、16 个关节目标的 PPO 独立计算和离线 JSONL replay，没有完整真机通信。不能加载本次 checkpoint，不能复用其归一化、手指协同和动作缩放。

### 5. 原仓库 deploy.py 的适用范围

当前训练仓库 `deploy.py` 是 GR1 示例，含 GR1 通信、旧关节转换和初始化姿态，不能直接用于 G1-D。

## 四、本阶段需要实现的内容

请建立独立、清晰、可迁移的部署目录，尽量复用现有策略和加载逻辑，不改训练算法、不重构训练主干，不修改原始数据和已训练权重。

### 1. 资料与现状梳理

列出有源码依据的结论和待实机确认事项，明确每份参考资料可复用的部分。

### 2. 显式指定 checkpoint 的推理加载器

- 从该 checkpoint 配置恢复策略。
- 使用 EMA 和 checkpoint 内 normalizer。
- 权重缺失或不匹配时明确失败，不静默使用随机初始化。
- 支持可配置 CPU/GPU，不依赖训练输出目录的隐式查找。
- 推理不启动训练、WandB、DDS 或硬件控制。

### 3. 观测处理

实现与训练一致的图像处理、28 维状态组装、两帧历史及启动历史处理；说明启动填充策略及其与训练 padding 的关系。

### 4. 离线观测回放和目标记录

- 支持真实 JSON/JPEG 或 Zarr 输入，至少交付一条基于现有真实数据的完整可运行路径。
- 输出预测动作、实测状态、时间戳和必要元信息。
- 无需 DDS、相机服务、Vision Pro 或机器人即可运行。
- 如提供模拟机器人，明确模拟反馈的定义与局限，不将其当成真实闭环验证。

### 5. 可替换的相机与机器人适配接口

- 离线实现能实际运行，不只留空函数。
- 根据已有源码尽量写好硬件适配实现或明确可接入的位置。
- 不确定的硬件参数外置并说明依据。
- 默认离线，不因 import、`--help` 或测试初始化硬件。
- 不要求部署侧运行 Vision Pro 或遥操重定向；当前策略直接输出关节目标，FK/IK 不是策略输入输出转换的必经步骤。
- 机械臂控制接口若需要前馈力矩或重力补偿，应根据现有实现调查并明确处理方式，不能随意以全零替代而宣称等价。

### 6. 动作队列与调度

- 清楚区分一次推理得到的 15 步、每步目标间隔及底层发送频率。
- 提供可配置的每次执行步数与重新推理逻辑，不把“每次必须执行全部 15 步”写死为唯一方案。
- 先建立简单可解释的方案，不盲目增加复杂异步系统。
- 记录推理耗时，明确推理延迟、反馈新鲜度和过期目标的处理。
- 不用无间隔快速发送 15 个目标代替约 30 Hz 执行。
- 避免每次推理都重置随机种子从而无意改变运行行为；固定种子用于可复现验证。

### 7. 为后续实机保留必要控制

- 明确启动、暂停、停止状态。
- 拒绝非有限数和失效反馈。
- 限位和限速依据可追溯，区分原始预测与实际发送目标。
- 区分数据集动作范围、URDF 限位与硬件已验证限制，不把它们当作同一概念。
- 退出、异常和初始化行为清楚说明，不擅自自动回零或抓取。
- 不控制策略之外的躯干、底盘等部位；调查复用控制器是否会附带设置这些目标并如实说明。

### 8. 迁移说明

交付依赖、文件清单、路径配置、离线运行命令及之后的实机联调顺序。优先给出可直接迁移使用的配置，不把本机绝对路径散落在代码中。

## 五、离线验证要求

用正式训练 checkpoint 和真实数据验证，不能只测合成模型：

- 检查权重、EMA、normalizer 加载成功，输出形状和数值有限。
- 核对部署预处理与原转换流程一致。
- 在相同输入、随机种子和随机状态下，与原策略推理路径比较。
- 保留随机增强与扩散采样；不要用不同随机状态的输出差异误判加载错误。
- 确认 15 步切片、两帧时间顺序和 episode 边界处理。
- 报告与演示 action 的整体及分部位误差，可检查连续输出的跳变；不要要求预测与演示完全一致。
- 演示来自训练集，离线误差与回放不证明真机闭环成功、泛化能力或模型充分利用了图像。
- 离线回放不能模拟真实执行反馈，也不能验证硬件动力学。
- 测量可用设备上的推理延迟；不能据此宣称目标真机主机已经达到实时要求。
- 覆盖会影响执行正确性的关键逻辑，避免无关的大量测试和框架。
- 保存实际命令、选用 checkpoint 与数据、配置和结果，检查产物与正式权重分开存放。

优先用现有 idp3 环境：

`/home/dingzirong/miniconda3/envs/idp3/bin/python`

实际路径、依赖和 GPU 可用性请现场检查。环境受限要如实报告，不伪造通过。按工具权限要求处理必要的环境与 GPU 检查，不为绕过限制改用其他执行方式。

## 六、必须留到实机阶段的内容

包括：

- 实际设备与 SDK 版本。
- 网卡与相机服务地址。
- 关节映射、方向、零位和 `thumb_0` 硬件坐标确认。
- 相机右目视野、颜色及传输延迟核对。
- 控制权限、模式与非策略关节保持方式。
- 增益、重力补偿与实际跟踪效果。
- 急停、暂停、停止及异常退出行为。
- 小幅动作测试、目标主机实时性、延迟与闭环任务效果。

能够根据源码离线完成的代码和检查请先做完；这些实机待办不应成为只交文档、不实现离线部分的理由。不要启动任何真实运动。

## 七、最终交付与执行方式

最终交付：

- 可运行的离线部署代码和示例命令。
- 用正式权重及真实数据生成的检查报告。
- 迁移到机器人主机需要的代码、依赖与配置清单。
- 清楚区分“已实现”“已离线验证”“尚待实机验证”的部署说明。
- 一份独立的部署进度记录，便于后续新对话继续。

请先读相关说明和代码、检查仓库状态，再直接实施。涉及常规可逆实现选择自行判断；只有确实无法从代码或数据确定且会影响实现的事项再询问，同时继续独立的离线工作。

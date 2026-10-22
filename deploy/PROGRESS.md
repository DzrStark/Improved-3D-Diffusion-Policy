# 独立部署进度记录

更新时间：2026-10-22。任务来源是同目录 `CODING_AGENT_PROMPT.md`，该文件保留原样。全部改动和检查产物只新增于本项目 `deploy/`，未改训练主干、正式权重、原始/转换数据或外部数采、部署参考和URDF目录。未连接机器人、DDS、相机服务，未启动遥操或真机程序，没有发布运动命令。

## 已实现

本次迁移复现的实测范围和剩余条件见 [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md)。

2026-10-22 发布前复现检查补充：转换工具已移入受 Git 跟踪的训练 `tools/`，verify 已跟随新路径；训练、转换及核对的默认数据位置统一为 `tsingtec_training_data/right_pick`，训练入口改用当前环境解释器。`goal.md` 更新当前状态并取消忽略。复制数据后核对器允许源根目录变化，仍逐条检查 JSON 哈希及数据内容。全新克隆说明见 `../REPRODUCING.md`。以上是首次离线交付之后的补充，不改变正式权重和原始/转换数据。

- `g1dp/policy.py`：显式checkpoint+训练源码路径，恢复原policy配置，严格恢复正式EMA/normalizer，拒绝缺失或不匹配，CPU/指定CUDA设备，不依赖训练目录隐式扫描或R3M预训练缓存下载。
- `observations.py` / `sources.py`：raw JSON/JPEG及Zarr输入、RGB处理、28维实测组装、两帧旧→新历史和首帧padding、episode复位、转换源JSON哈希和原始时间恢复。
- `replay.py`：真实离线观测回放和目标JSONL、原16步/切片15步、时间/延迟/实际状态、配置/命令/哈希/分组误差记录。数据反馈外生，不是物理模拟。
- `control.py` / `session.py`：同步队列、E=1..15、30Hz目标时间、过期丢弃/重推理、新鲜度/时钟/非有限拒绝、pause/start/stop、基于有来源边界的位置/限速过滤、异常退出关闭。
- `adapters.py`：可运行记录sink、相机与机器人协议、显式注入ImageClient、仿射硬件边界、无发布Dex3消息构建、动力学力矩provider接入点。没有可直接打开真机的DDS transport。
- `configs/`：四任务相对路径配置、默认CPU/Zarr，以及未启用、参数留空的硬件核对模板。
- `verify.py` / `tests/`：正式权重和真实数据检查、原策略对照、关键执行逻辑测试。
- `README.md` / `SOURCE_AUDIT.md` / `MIGRATION.md` / `OFFLINE_REPORT.md`：契约、源码证据、迁移命令/文件/依赖、真实结果、实机顺序和边界。

## 已离线验证

解释器：`/home/dingzirong/miniconda3/envs/idp3/bin/python`。当前权限环境CUDA不可用，没有提权或绕过探测限制，CPU四线程完成验证。

1. 核实四任务latest、epoch300、Hydra配置和日志，末轮日志均epoch300，latest计数epoch301。任务权重未修改。
2. 四任务正式EMA+真实数据均输出有限15×28/16×28；right_pick另跑raw JSON/JPEG入口。全部命令、源路径和结果在各`summary.json`中。
3. right_pick全部60条、10,344帧raw state/action与Zarr一致；180张RGB与原转换器逐像素一致；全部10,404窗口不跨episode；首尾/启动历史与原Sampler+Dataset一致。
4. 同输入及完整随机状态，与原生pretrained R3M加载后恢复同一EMA的原策略比较，完整与切片最大差均0；保留随机增强/扩散。normalizer已完整恢复且范围与数据核对一致。
5. 9项执行逻辑测试、编译、导入和两个CLI --help通过；离线模块无SDK/相机/遥操/WandB导入或初始化。
6. 实际延迟/整体及分部位误差、跳变、全部时间戳和边界检查已保存于`reports/`，摘要在`OFFLINE_REPORT.md`。这些是训练集离线证据，不证明硬件闭环、泛化或图像使用充分。

继续复现时从项目根目录运行README中的命令，**选新的输出目录**。已存报告目录会拒绝覆盖。部署加载器无需`~/.r3m`；原生对照验证需要已有cache，绝不静默下载。

## 关键发现，后续不要丢失

- 左手DDS次序是thumb0/1/2、middle0/1、index0/1；右手是thumb0/1/2、index0/1、middle0/1。臂候选DDS编号15..28，顺序肩pitch/roll/yaw、肘、腕roll/pitch/yaw。
- right_pick全部JSON的joint_names为空、hand_coordinates缺失；不能离线确定旧数据硬件零位。旧数据保持原值，不能套文档里的thumb_0加30°或缩放。现有文档与当前live代码也不完全一致。
- 所给G1-D URDF28个候选关节名都存在，但与G1_29全身控制器不等价。right_pick实测state无越界，演示action有38,791个手部数值超URDF限位；不将这些来源混作硬件确认值。
- 原G1_29控制器初始化就锁定/发布非臂关节及启动线程；Dex3控制器也会自动发布。不能直接实例化它们做离线检查或作为joint-only transport。
- 数采臂目标同时带`rnea(q,0,0)`前馈，部署不能随意用全零替代。策略本身无需IK，但力矩计算仍需经确认模型。
- 当前CPU一次推理约160–180ms，E=4时间窗不足；Session会丢弃过期槽位。离线回放忽略推理成本推进虚拟时间，不能当实时性能或闭环。
- 录制时间不是传感器源时间，ImageClient读取缓存也不能刷新其时间戳；硬件层必须提供同帧/同组状态的真实接收时间。

## 尚待实机阶段

从 `MIGRATION.md` 的联调顺序继续：设备/SDK/网络、硬件映射/方向/零位/thumb_0、相机颜色视野与时间、控制权限及非策略关节、joint-only transport/看门狗、硬件已验证限位限速增益、动力学模型与前馈、急停暂停停止异常退出、小幅动作和目标机实时性，最后闭环任务。

这些硬件事项没有伪造完成值；配置模板保持disabled/null。若后续只需继续离线分析，可直接复用现有入口和正式checkpoint，无需重训或启动任何硬件控制器。

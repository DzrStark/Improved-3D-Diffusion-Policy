# G1-D + 双 Dex3：RGB Diffusion Policy 离线部署

全新克隆先阅读 [REPRODUCING.md](../REPRODUCING.md)：G1-D 数据和正式权重需另行获取。下列命令使用已安装依赖的 Python 环境；完整 verify 还需要已有 R3M 缓存和对应 URDF。

本目录已实现并用四个任务的正式 `latest.ckpt` 和真实数据跑通离线推理。完整核对优先使用 right_pick。**没有连接机器人、相机服务或 DDS，没有发布电机命令。** 未修改训练算法、原始数据、正式权重或外部参考目录。

## 运行

从项目根目录运行（即包含 `deploy/`、`tsingtec_training_data/` 的目录）：

```bash
PY=python
$PY -m deploy.replay --config deploy/configs/right_pick.json \
  --output deploy/reports/my_right_pick --episodes 0,1,59 --max-frames 32

# 直接读取原始 JSON/JPEG；每个目标后重新推理
$PY -m deploy.replay --config deploy/configs/right_pick.json --source raw \
  --execute-steps 1 --output deploy/reports/my_raw --max-frames 8

$PY -m unittest discover -s deploy/tests -v
$PY -m deploy.verify --config deploy/configs/right_pick.json \
  --urdf /path/to/g1_d.urdf --output deploy/reports/my_verification
```

输出目录必须不存在。`--max-frames 0` 回放选定 episode 全长，`--episodes` 是按原始数字编号排序后的 **零基 episode 索引**，例如 right_pick 的索引 0 是 episode_0001。其他三任务只替换配置文件：`right_place.json`、`stack.json`、`right_to_left_handover_low.json`。`--device cuda:0` 显式选择可用设备，不自动回退 CPU。

四份配置的路径均相对 `path_base` 解析；默认是项目根目录，迁移只需修改配置。checkpoint 和数据必须显式给出，加载器不会扫描目录选“最新”。原训练主干不需要部署启动脚本、Vision Pro、遥操或 FK/IK。完整迁移说明见 [MIGRATION.md](MIGRATION.md)。

当前idp3的实际版本、安装/打包方法和真机依赖缺口见 [IDP3_ENVIRONMENT.md](IDP3_ENVIRONMENT.md)，新建部署环境配置为 [environment-idp3.yml](environment-idp3.yml)。

四份默认配置中的 `raw_root` 均已填写，所以默认 Zarr 回放也需要原始 JSON。只有 Zarr 时可将 `raw_root` 设为 `null`，此时使用名义 30 Hz，不能声称复现原始采样时间。`verify.py` 的完整数据检查针对 right_pick；其他任务已做回放，但未完成相同范围的全量核对。

本开发机 URDF 现位于 `../dynamimic-deploy/reference/assets/urdf/g1_d_description/g1_d.urdf`，哈希与历史报告一致；原 `../g1_d_description/` 已失效，其他主机需单独准备对应文件。

## 输入和恢复契约

- 输入是两帧从旧到新的右目全视野 RGB、对应的两帧实测关节位置；状态顺序固定为左臂7、右臂7、左手7、右手7，单位 rad。
- 原始相机双目 BGR 先取右半幅，再 BGR→RGB、OpenCV `INTER_AREA` 缩至 224×224。原始 `color_1.jpg` 已是右目，不能再取半幅。Zarr 已经是 RGB 224×224，不交换通道。
- 网络输入 float32、0–255。policy 内部除以255，R3M 内部 ImageNet 标准化；部署不重复处理。保留原 RandomCrop、RandomRotation、ColorJitter 和扩散采样，`eval()` 不关闭这些随机变换。
- 启动历史为 `[第一帧, 第一帧]`，与训练 `pad_before=1` 一致；第二次为 `[第一帧, 第二帧]`。episode 切换清空历史、队列。历史只使用实际观测，不用策略目标。
- `horizon=16, n_obs_steps=2, n_action_steps=16`；原实现 `action_pred[:,1:16]` 实际返回 **15×28**。第一个返回目标对齐最新观测帧，之后间隔名义 1/30 s。全部是绝对关节位置。
- 从可信的训练 checkpoint 配置实例化原 `DiffusionImagePolicy`，严格恢复 EMA 的289个张量及 checkpoint 内 normalizer。state/image 是 identity，action 是原 limits；不调用 Dataset 重新估计。
- 部署构造 R3M 时只创建相同 ResNet18/BatchNorm 架构，保留 `.module` 权重命名，由正式 EMA 提供全部参数和 buffer，避免下载或依赖用户的 `~/.r3m`。作用域限定的 factory 不改训练文件。原策略对比另走原生 pretrained 加载路径，已证实输出完全一致。

## 回放记录与调度

`targets.jsonl` 每帧记录实测关节、演示 action、episode/帧索引、历史顺序、时间戳；推理帧额外记录完整16步和切片15步、推理耗时；被队列选中的目标记为 `recorded_target`。`actual_sent_target` 始终为 null。`summary.json` 保存实际命令、有效配置、权重 SHA-256、误差和耗时。

回放使用录制时间推进**虚拟时钟**，不按墙钟睡眠，也不向设备发送命令。推理耗时单独测量，但不从虚拟时间扣除。因此它能复查输入、动作对齐和队列逻辑，不能当作实时执行。`RecordingRobot` 的反馈始终来自数据集，写入目标不会改变下一次反馈；它不模拟物理或闭环。Zarr 没有保存时间戳，提供 `raw_root` 时通过转换报告和全部 JSON 哈希核对后恢复录制时间；不提供时明确标记为名义时间，不能冒充实测采样时刻。

`ActionQueue` 每次选 `execute_steps=1..15` 个目标，默认4，用完重推理。目标时间为最新观测时间 `+ k/30`；迟到超过 `max_lateness_s` 的槽位丢弃，绝不补发一串目标或把它们统一延后。数据实际录制间隔不完全等于1/30，某个回放观测时刻可能尚无目标到期，记录仍保留该帧。

后续接硬件可复用 `Session.tick()`：启动保持 paused，显式 `start()`，按名义30 Hz 调用；每次用相机接收时间和反馈接收时间检查新鲜度/偏差，推理后再次读取实际反馈。过期目标跳过，过期 chunk 清空等待新观测重推理；异常进入 stopped 并关闭适配器。暂停清空队列，恢复后重新获取历史。不每次推理重置种子；固定种子只在运行开始或对齐验证时设置。

底层发送频率是 transport 的独立职责。参考旧控制器臂250 Hz、手100 Hz，不能把它们当成策略频率，更不能复制另一个 PPO 项目的60 Hz。`Session` 没有隐含发布线程；底层持续保持、看门狗、暂停/断联释放策略仍需在真机控制权限确认后实现并验证。停止发布不等于物理急停，退出不自动回零或抓取。

## 控制边界与硬件接入状态

`SafetyFilter` 拒绝非有限数、失效反馈和时钟错误；同时约束位置、相邻发送目标增量和目标相对反馈的偏差，约束交集为空则拒绝。长暂停不会累计成一次大跳变。原始预测与过滤后发送目标分别记录。该过滤器的单元测试使用明示的测试参数，**当前未选定硬件安全限位或限速**；数据动作范围、URDF参考值和硬件确认值分别见 [limits_comparison.json](reports/right_pick_verification_final/limits_comparison.json)。

`JointOnlyRobot` 已实现可注入 transport 的读写与28维仿射坐标映射，要求显式验证标志、只覆盖策略关节的 transport 和14维力矩 provider。`dex3_command()` 可按原 DDS 顺序构建七电机消息，但不创建 publisher。`gravity_provider()` 用经过核对的 Pinocchio 模型和实测非臂姿态计算 `rnea(q,0,0)`，无需求 IK。均未在硬件验证；不以全零力矩声称与原数采等价。

**尚未交付可直接启动的真机 DDS transport。** 原 `G1_29_ArmController` 初始化即发布全身保持目标，不符合本阶段只操作策略关节的边界，不能直接注入；其初始化零目标、控制模式及线程生命周期也需要重新审查。相机可注入现有 ImageClient，但其缓存帧的读取时间不等于接收时间；必须给适配器提供可靠接收回调时间戳后再接 Session。上述明确接入点、源码证据和待确认项见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)，外置核对表见 [hardware_review.template.json](configs/hardware_review.template.json)。

## 已验证与尚待验证

实际结果见 [OFFLINE_REPORT.md](OFFLINE_REPORT.md)，独立续接记录见 [PROGRESS.md](PROGRESS.md)。已完成四任务正式 EMA 推理、right_pick 原路径一致性、真实预处理/边界/normalizer 检查及关键执行逻辑测试。

尚待实机：设备与 SDK 版本、网络和相机地址、关节顺序/方向/零位与 thumb_0 坐标、G1_29和G1-D的接口差异、控制权限和非策略关节归属、已验证限位/限速/增益、重力补偿、急停/暂停/异常退出、目标主机实时性和闭环任务效果。训练集误差不能证明泛化、充分使用图像或真机成功率。

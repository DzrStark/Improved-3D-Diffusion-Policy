# 源码与资料核对（2026-10-22）

以下路径以项目根目录或相邻 `../data_collection` 为基准。实际读取文件的 SHA-256 已保存在 [source_manifest.json](reports/source_manifest.json)。只静态阅读硬件源码，没有导入或实例化控制器。

2026-10-22 路径更新：转换器已迁入训练代码 `tools/`；下表 URDF 及历史清单中的旧绝对路径保留作记录。当前 URDF 为 `../dynamimic-deploy/reference/assets/urdf/g1_d_description/g1_d.urdf`，SHA-256 仍为 `b03f950227c03ff0265477fb5a8dcdd21293e927f8513386a792fcc8bb387cfb`。

| 资料 | 可复用的证据 | 本次边界 |
| --- | --- | --- |
| `Improved-3D-Diffusion-Policy/goal.md`、`tools/step2_config_logic.md`、`step3_ema_logic.md`、`train_g1_dex3_rgb.md` 及检查代码 | R3M、两帧、padding、EMA、28维、随机增强契约 | 训练已完成；四个输出日志末尾均为 epoch300，当前状态见 goal.md 顶部 |
| `.hydra/config.yaml`、checkpoint 的 `cfg` | 16/2/16、DDIM 10次推理、50训练时刻、ResNet18、增强、双臂双手维度 | checkpoint 内配置才是恢复依据；不启动 workspace、optimizer 或 WandB |
| `diffusion_policy_3d/policy/diffusion_image_policy.py::predict_action` | 归一化、图像除255、HWC→CHW、原始16步、从索引1开始切片 | 保留实际15步行为；无增量积分和动作缩放 |
| `dataset/gr1_dex_dataset_image.py`、`common/sampler.py` | NHWC float32、state identity、action limits、episode内首尾重复padding | 部署首帧重复一次，不将策略目标当反馈 |
| `model/vision/timm_obs_encoder.py`、`third_party/r3m/r3m/models/models_r3m.py` | eval时仍走随机增强；R3M内部标准化 | 不添加第二遍图像标准化，不关闭增强 |
| `Improved-3D-Diffusion-Policy/tools/convert_unitree_to_zarr.py` | `colors.color_1`，OpenCV BGR→RGB、完整视野 INTER_AREA 224×224 | raw 与 Zarr 两条读数路径分别处理；无裁切/PIL/JPEG重编码 |
| `../data_collection/run/scripts/teleop_dex3_capture.py` | Head BGR 右半写 color_1；`get_current_dual_arm_q` 和 hand_state 写 states；IK target 和 hand_action 写 actions | 只能确认现有采集代码语义；不能据此断言旧 episode 的硬件版本和零位 |
| `run/scripts/dex3_capture_recording.py`、`teleop/utils/episode_writer.py` | writer在add_item入口记录monotonic/wall时间；原样保存states/actions；异步写图 | 时间戳不是相机曝光或编码器采样时间；资料明确缺少源传感器时间戳 |
| `robot_control/robot_arm.py::G1_29_JointArmIndex` | 左臂15..21、右臂22..28；肩pitch/roll/yaw、肘、腕roll/pitch/yaw | 是G1_29源码候选映射，不是已验证G1-D电机映射 |
| `robot_hand_unitree.py::*JointIndex` | 左手thumb0/1/2、middle0/1、index0/1；右手thumb0/1/2、index0/1、middle0/1；直接DDS q(rad) | 左右手组内次序不同，符号不翻转，不按对称手假设统一排序 |
| `run/scripts/evaluate_place_live.py`、`dex3_live_support.py` | Joint28的接口、消息构造、参考姿态和可视化组织方式 | 旧模型是单帧/16步/语言等不同契约，不复用服务、裁切、JPEG95或初始化动作 |
| 两份 `docs/README_dex3_*.md` | 历史有原始透传、加减30°、`legacy/2+30°` 多种方案 | 当前 `dex3_live_support.HandBackend` 实际是透传，与文档最新映射不一致；不按文档或最新代码给训练数据自动补偿 |
| `../g1_d_description/g1_d.urdf` | 包含与候选顺序同名的28个臂手关节，支持导出参考位置/速度限位 | 该描述的躯干是Yaw_Joint/torso_Joint；G1_29另有腿和waist等，不能仅因臂关节同名证明控制接口等价 |
| `../dynamimic-deploy/DEPLOYMENT.md` | 独立JSONL记录的组织方式 | PPO为102观测、10动作、16关节、60Hz，无本次权重/完整通信；不复用其归一化、手指协同或动作缩放 |
| 训练目录 `deploy.py` | 作者GR1示例 | 会引入GR1通信、旧关节转换和初始姿态，不调用 |

## 数据坐标的证据强度

right_pick 的60条 JSON 已逐项与转换数据核对：全部 `joint_names` 为空，全部没有 `info.hand_coordinates`。因此能够证明28维拼接、原值、符号和图像转换未改变，却不能独立证明旧数据的电机编号/硬件版本/零位。候选关节名顺序来自采集控制器枚举，并在输出中明确标成 candidate。

部署默认保留数据坐标。若新硬件需要仿射转换，应在 **硬件边界** 明确设置 `hardware = training * scale + offset`，反馈用逆变换。不得因为新文档出现 thumb_0 偏置就重写旧数据、normalizer或checkpoint。若硬件数组顺序不同，先在 transport 内按已核实名称重排到候选28维顺序。

本次核对：URDF能找到全部28个候选关节；right_pick全部实测state在该URDF范围内，而演示action有38,791个分量越界，均在手部。原因和硬件实际可接受范围不能仅从这些文件确定，不归因为某种补偿版本。完整逐关节计数、数据范围、URDF范围/速度与硬件null值分别存入 `limits_comparison.json`。

## 控制器副作用和力矩

`G1_29_ArmController.__init__` 先设14维q/tau为零，创建DDS publisher/subscriber，等待反馈后创建全身LowCmd，给非臂关节设置增益并将其q保持在启动实测位置，随后立即启动250Hz发布线程。它不是只读状态提供器，也不是仅控制28维的透明封装。`motion_mode` 还涉及不同topic及控制权字段；`ctrl_dual_arm_go_home()` 会回零，且发布线程没有本项目可调用的完整关闭接口。不能直接复用它满足“不控制躯干/底盘”的约束。

Dex3 控制器构造时创建订阅线程和控制子进程，目标起始为零，默认100Hz持续发布，等待反馈的判据还使用 `any(state)`；不能用实例化做离线检查。可复用手部DDS消息契约：每手7电机、mode=`id | 0x10`、显式q/kp/kd/dq/tau；SDK消息的构建已封装为无发布函数。

数采不是向机械臂只发送位置：`G1_29_ArmIK.solve_ik` 在目标q上将速度置零，再调用 Pinocchio `rnea(q,0,0)` 产生前馈力矩，`ctrl_dual_arm(target,tau)` 同时传给底层。部署直接预测关节q，无需IK，但仍需经过核对的重力模型、姿态和惯性参数提供tau。`gravity_provider()` 给出此计算接入点；不加载不匹配模型，不在本阶段假设全零等价。实际增益、力矩与跟踪需要真机确认。

## 下一阶段应接的位置

1. 经运控确认提供 **joint-only** transport：`read() -> Feedback(q28, 接收monotonic秒)`；`write(arm14, tau14, hand14, now)`；`stop()` 停止本程序发布。DDS低层topic、控制权、未控制关节归属和看门狗必须先确认，不能自动切换Debug模式或套用全身锁定。
2. transport按硬件已验证关节名排序，并提供三组反馈（臂、左手、右手）的真实接收时间及同步策略；返回的聚合时间不能隐藏其中任意过期一路。具体聚合/最大偏差需在硬件核实。
3. 相机显式连接后注入 `ImageClientCamera`。接收时间回调必须对应**同一帧**，不能取“最新接收时间”给旧缓存帧盖章；应在接收线程原子保存图像及时间。现有ImageClient没有这里可直接使用的可靠源时间字段，需在目标机接收层补齐。
4. 填写硬件核对表并注入 `JointOnlyRobot`、已确认SafetyFilter和力矩provider，再使用 `Session`。启动保持paused；外部显式start，30Hz调度tick，异常finally close。底层持续发送频率、保持和释放由transport承担，不能把Session当作完整硬件驱动。

需要实机确认：实际设备/SDK、网卡和相机地址、关节映射方向/零位/thumb_0、图像颜色视野及延迟、非策略关节与控制模式、增益/重力/限速/跟踪、急停暂停退出、小幅动作、目标主机实时性及闭环效果。这些不确定项已外置，离线部分不依赖它们。

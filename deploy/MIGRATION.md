# 迁移与后续联调

## 文件和依赖

迁移以下内容并保持相对结构，或修改 `configs/*.json` 的路径：

| 内容 | 必需程度 |
| --- | --- |
| `deploy/` 的代码、配置、说明 | 必需；报告可选择保留作基线 |
| `Improved-3D-Diffusion-Policy/diffusion_policy_3d/` | 必需；复用原policy、U-Net、normalizer、encoder等，不启动train.py |
| `Improved-3D-Diffusion-Policy/tools/` | 数据转换及完整 verify 需要；转换器已从原始数据目录迁入此处 |
| `third_party/r3m/r3m/` | 必需；使用本仓库修改过的R3M，不能换未经比较的PyPI版本 |
| 选定任务的正式 `latest.ckpt` | 必需；约1.44GB，完整训练包含model/EMA/optimizer，部署只取EMA |
| 对应 `tsingtec_training_data/<task>` | 离线回放必需；保持Zarr目录完整，含conversion_report.json |
| 对应原始JSON/JPEG | raw回放与严格核对需要；Zarr带实测时间戳时需要原始JSON和哈希，不必读取JPEG |
| `g1_d.urdf` | 限位参考核对需要；部署推理不需要FK/URDF |
| `~/.r3m/r3m_18/{config.yaml,model.pt}` | 仅 `verify.py` 的原生加载对照需要；正式部署加载器不需要它们 |
| Unitree SDK、teleimager、Pinocchio及经过确认的模型 | 后续硬件适配才需要，离线回放不导入SDK/相机 |

已验证环境为 Python3.8、PyTorch2.1.0+cu121、torchvision0.16.0+cu121、OpenCV4.5.3、Zarr2.12.0。精确版本见 `requirements-offline.txt` 和 [environment-freeze.txt](reports/environment-freeze.txt)。后者是完整参考环境快照，含本机editable路径，不能直接当作跨主机可安装锁文件。无需为本部署安装WandB、VisionPro、StarVLA、transformers或仿真平台。

建议优先复制/打包现有idp3环境，再复制对应源码和权重。重建环境时先安装匹配目标CPU/GPU驱动的torch/torchvision wheel，然后安装离线requirements并保留本地R3M源码。示例（在允许安装依赖的目标主机执行，本阶段未安装或改环境）：

```bash
# 先准备torch/torchvision wheel及依赖wheelhouse，再在项目根目录：
python -m pip install --no-index --find-links /path/to/wheelhouse \
  -r deploy/requirements-offline.txt
```

`omegaconf`、Hydra、diffusers、huggingface-hub、timm之间需要保持本环境兼容版本，不能随意升级最新包。当前文件记录的是已安装版本而不是声称从零安装已验证。加载完整训练包使用dill/pickle，只加载自己可信的正式文件；SHA-256用于迁移核对，不是陌生pickle的安全隔离。

## 路径与离线验收

Git 不包含数据、正式权重、R3M 预训练缓存或设备 URDF，获取和布局见 [REPRODUCING.md](../REPRODUCING.md)。旧报告中的绝对路径是原机记录，不能直接照搬。

配置中 `path_base: ../..` 相对配置文件位置，默认定位仓库根目录。变更 `training_root`、`checkpoint`、`zarr`、`raw_root` 即可迁移；代码没有散落本机工作目录路径。输出目录用CLI显式给出，与正式权重分离。

```bash
cd /path/to/Improved-3D-Diffusion-Policy
python -m deploy.replay --help
python -m unittest discover -s deploy/tests -v
python -m deploy.replay --config deploy/configs/right_pick.json \
  --output /path/to/new_report --max-frames 32
# 有可用GPU后显式选择；无GPU时会报错而不是回退
python -m deploy.replay --config deploy/configs/right_pick.json --device cuda:0 \
  --output /path/to/new_gpu_report --max-frames 32
```

先核对权重SHA-256、EMA/normalizer/形状/有限值，再用相同输入和完整随机状态与原路径比较。版本或设备变化允许合理数值容差，应保留新报告，不用不同随机增强/扩散噪声的结果判断恢复失败。生产推理一次设种子；随机增强继续保留。

回放是虚拟时间，耗时报告不意味着30Hz执行。当前CPU每次推理约160–180ms，默认4步覆盖的时间不足，Session会丢弃已过期槽位。应在目标机实测后选择每轮执行步数/推理设备，核对延迟、新鲜度和超时；不要简单压缩发送间隔以追赶计划。本实现先采用同步、可解释调度，未构造异步推理系统。

## 真机阶段顺序（本阶段均未执行）

1. 确认实际G1-D/SDK与G1_29参考接口差异、控制权限及非策略关节归属。只读核实设备状态和网络，先不要创建旧控制器。完成 `hardware_review.template.json`，保留校验依据与确认人。
2. 根据单关节只读反馈和机械标定核实名称、电机ID、左右手符号、零位和thumb_0坐标；确认训练坐标与现硬件的映射。核对28维范围和URDF差异，填写硬件已验证位置/速度范围与增益。
3. 只接相机/状态订阅，核对右目全视野、颜色、分辨率和接收时间戳；三路关节反馈均需新鲜。录制观测并离线推理，不发送命令。
4. 运控侧实现/审查joint-only transport、模式/控制权/看门狗、暂停保持/释放、停止、异常退出与急停。确认它不改躯干、底盘或腿目标；不能直接包裹会锁定全身的旧G1_29控制器。
5. 用匹配设备的动力学模型和实测姿态验证重力前馈；核对参考 `rnea(q,0,0)` 与实机所需力矩、增益和跟踪。无需为策略额外求IK。
6. 明确现场急停和支持条件后，先做小幅、限速的单组动作，逐项验证start/pause/stop、反馈中断、相机断流和异常退出，再连接策略。退出不自动回零或抓取。
7. 在目标主机测推理/接收/发送延迟和反馈时效，选择E步和发送频率，确认无过期目标突发发送，最后评估实际闭环任务。保存原预测、过滤后目标、实际反馈与全部时间戳。

本阶段没有硬件验证结果，当前进度不能解释为已具备直接启动真机的条件。

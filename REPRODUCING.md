# G1-D + 双 Dex3 RGB：复现入口

本文命令从 Git 仓库根目录执行，即同时包含 `deploy/`、`third_party/` 和内层
`Improved-3D-Diffusion-Policy/` 的目录。内层目录是训练代码根目录。
上游 README 的 Google Drive 数据、GR1 启动脚本和 RealSense 示例属于作者原项目，
不能替代本分支的 G1-D 数据、28 维正式权重或硬件驱动。

## 1. 先准备仓库之外的产物

Git 只包含代码、配置、说明和已有验证报告；数据、权重、URDF 和 R3M 缓存没有随仓库发布。
截至 2026-10-22，本仓库没有提供 G1-D 数据及四任务正式权重的公开下载地址。
复现者需先向产物提供者获取它们；仅 clone 无法重现正式推理结果。

| 内容 | 默认位置（相对仓库根目录） | 用途 |
| --- | --- | --- |
| 原始 JSON/JPEG | `tsingtec_raw_data/` | 转换、raw 回放、完整核对 |
| 完整 Zarr v2 store | `tsingtec_training_data/<task>/` | 训练、Zarr 回放；包含 `conversion_report.json` |
| 正式 checkpoint | `Improved-3D-Diffusion-Policy/data/outputs/g1_dex3_rgb/<run>/checkpoints/latest.ckpt` | 已训练策略回放；四个确切位置见 `deploy/configs/*.json` |
| 原生 R3M ResNet18 的 `config.yaml` 和 `model.pt` | `~/.r3m/r3m_18/` | 从头训练初始化、原生策略对照；部署加载器本身不需要 |
| 设备对应的 `g1_d.urdf` | 用 `deploy.verify --urdf` 显式指定 | 完整核对中的限位参考；普通回放不需要 |

数据目录保持 `episode_*/data.json` 和 JSON 中图像相对路径的结构。完整核对需要
right_pick 全部 60 条、10,344 帧以及 `data.json` 中的 `info.joint_names`；普通回放的
数据要求见 `deploy/README.md`。R3M 原生加载器在缺缓存时尝试联网下载，离线训练前必须准备缓存。

四任务正式 `latest.ckpt` 的 SHA-256 基线来自已提交的回放报告：

| 任务 | SHA-256 |
| --- | --- |
| right_pick | `f5d52bf2431bf7df5b0e7b03f6936f6578af176c68f664d99c23a75a38d75b05` |
| right_place | `f57bf57b34a2efd288ef84bb1851178529f817e92844de3dea60bb92af7070b6` |
| stack | `3f9ab3ff3fe424cc234b173e89b566a4296bf3a27da6a7495956100f13488f53` |
| right_to_left_handover_low | `3de90da5d97f8246d45906edd277e7b492b574528e1743234ebc49c19222dae2` |

## 2. 环境

部署环境按 `deploy/IDP3_ENVIRONMENT.md` 重建，在新环境中操作：

```bash
conda env create -f deploy/environment-idp3.yml
conda activate idp3-deploy
python -m pip check
```

这份环境配置记录了已有 Python 3.8 环境的版本，尚未验证从零联网重建。
需核对目标主机 Python/系统架构、Torch/CUDA 构建和 GPU 可用性；ARM/Jetson 不能照搬 x86_64 wheel。

若需要训练或运行训练检查器，还要安装训练依赖和本仓库 R3M：

```bash
python -m pip install -r Improved-3D-Diffusion-Policy/requirements-g1-rgb.txt
python -m pip install --no-deps -e ./third_party/r3m
python -m pip install --no-deps -e ./Improved-3D-Diffusion-Policy
python -m pip check
```

部署 requirements 不含训练 workspace 导入的 WandB、pandas；不要将部署环境等同于训练环境。
不要直接用 `deploy/reports/environment-freeze.txt` 安装，它含原机 editable/临时 wheel 路径。

## 3. 数据转换

全部七组原始数据都已准备、且各输出目录尚不存在时：

```bash
python Improved-3D-Diffusion-Policy/tools/convert_all_unitree_to_zarr.py
```

只准备 right_pick 时，使用单组入口，默认输出与当前训练/部署配置一致：

```bash
python Improved-3D-Diffusion-Policy/tools/convert_unitree_to_zarr.py
python Improved-3D-Diffusion-Policy/tools/check_unitree_zarr.py --all-images
```

显式路径通过单组入口的 `--raw-dir`/`--output`、核对入口的 `--raw-dir`/`--zarr`，
或批量入口的 `--raw-root`/`--output-root` 指定。已有 Zarr 无需重新转换，入口拒绝覆盖。
迁移后允许源目录绝对路径变化，核对仍检查原始 JSON 哈希、state/action、RGB 和 episode 边界。
旧 `Improved-3D-Diffusion-Policy/data/dex3_pick_right_rgb.zarr` 是首轮转换产物；
若使用旧路径，应显式指定，不需要重复存放同一数据。

## 4. 无需数据或权重的检查

```bash
python -m unittest discover -s deploy/tests -v
python -m unittest discover -s Improved-3D-Diffusion-Policy/tools/tests -v
python -m deploy.replay --help
python -m deploy.verify --help
```

转换测试使用临时合成数据，只检查转换/迁移逻辑，不是正式模型或真实数据验证。

## 5. 正式权重回放与完整核对

准备产物后，修改 `deploy/configs/right_pick.json` 中路径以匹配实际存放位置。
`path_base` 相对配置文件位置解析；CLI 的输出路径相对当前工作目录。

```bash
python -m deploy.replay --config deploy/configs/right_pick.json \
  --output /tmp/g1-right-pick-replay --episodes 0 --max-frames 8

# 还需完整 right_pick 原始数据、已有 R3M 缓存和 URDF：
python -m deploy.verify --config deploy/configs/right_pick.json \
  --urdf /path/to/g1_d.urdf --output /tmp/g1-right-pick-verification
```

每次使用尚不存在的输出目录。四份默认配置都填写了 `raw_root`，因此默认 Zarr 回放也会
读取原始 JSON 以核对哈希、恢复录制时间。只有 Zarr 时可将该配置值设为 `null`，
回放会明确使用名义 30 Hz 时间；不能据此声称复现了原始录制时序。
`deploy.verify` 依赖原始 JPEG 和原生加载路径，其当前数据检查针对 right_pick；
不能仅换成其他任务配置就宣称完成了同等完整核对。

同输入比较需恢复 Python/NumPy/Torch 随机状态；随机增强和扩散采样继续保留。
跨设备/版本不保证输出逐位一致，记录实际环境与误差。离线回放不模拟动作对下一观测的影响，
也不计入推理耗时对执行时钟的影响；它不证明真机闭环成功。

## 6. 训练

从仓库根目录可查看配置，再按实际 GPU 启动；第二条命令会执行完整训练：

```bash
bash Improved-3D-Diffusion-Policy/tools/train_g1_dex3_rgb.sh -- --cfg job --resolve

# 默认 WandB online，先在训练终端完成 wandb login：
bash Improved-3D-Diffusion-Policy/tools/train_g1_dex3_rgb.sh --gpu 0
```

默认使用当前 PATH 中的 `python`，可用 `PYTHON_BIN` 或 `--python` 覆盖。
`--zarr`、`--output-dir` 相对内层训练代码根目录解析。默认全部 right_pick 数据、301 epochs、
batch size 32、seed 42，无独立验证集。离线日志可在末尾加 `-- logging.mode=offline`。
其他三任务的参数覆盖示例见 `Improved-3D-Diffusion-Policy/tools/train_g1_dex3_rgb.md`。
2026-10-22 已清理的 7 个短程检查 checkpoint 不属于正式产物；重新运行检查器会再次产生权重。

真机通信与硬件映射尚未完成；后续接入说明见 `deploy/next_goal.md`。

# 当前 idp3 环境与迁移方法

本说明基于2026-10-22实际检查的idp3环境，适用于本项目当前 **G1-D + 双Dex3的R3M RGB Diffusion Policy**。已验证的是现有环境中的加载、离线回放和检查；本次没有安装、升级或重建环境，也没有连接机器人。

## 当前环境

| 项目 | 本机实测 |
| --- | --- |
| Conda环境 | `idp3` |
| Python | `3.8.20` |
| 解释器 | `/home/dingzirong/miniconda3/envs/idp3/bin/python` |
| 系统/架构 | Linux x86_64；详细平台见环境快照 |
| Torch运行版本 | `2.1.0+cu121` |
| Torchvision运行版本 | `0.16.0+cu121` |
| Torch CUDA构建版本 | `12.1` |
| 当前工具环境GPU可用性 | `False`，可见设备数0，NVML无法初始化 |
| pip / setuptools / wheel | `24.2` / `59.5.0` / `0.44.0` |
| `python -m pip check` | `No broken requirements found.` |

Torch的发行包元数据版本为`2.1.0`，运行时版本为`2.1.0+cu121`；Torchvision同理。因此配置中的版本号与运行时输出并不矛盾。安装了CUDA构建不代表目标机驱动、GPU和权限已经可用，迁移后必须重新检查。

## 文件怎么用

- [environment-idp3.yml](environment-idp3.yml)：按当前已验证版本整理的RGB部署环境重建配置，创建新环境`idp3-deploy`，不覆盖原`idp3`。覆盖推理、回放、原Sampler/Dataset检查，不包含整套历史训练/仿真软件。
- [requirements-offline.txt](requirements-offline.txt)：已有部署直接依赖版本清单。名称中的offline表示用于离线部署代码；执行pip安装是否联网取决于安装参数和wheel来源。
- [idp3_environment.json](reports/idp3_environment.json)：本次解释器、平台、实际版本、可用设备和可选硬件模块检查结果。
- [environment-freeze.txt](reports/environment-freeze.txt)：前次完整环境快照，包含很多当前部署不用的包和本机`file:///tmp/idp3_wheels/...`路径，不能直接复制为跨主机安装命令。

关键版本如下，完整部署列表见YAML：

| 功能 | 包及当前版本 |
| --- | --- |
| 策略网络/图像变换 | torch 2.1.0、torchvision 0.16.0、timm 0.9.7、einops 0.4.1 |
| 扩散采样 | diffusers 0.11.1、huggingface-hub 0.25.2 |
| 图像/数组 | numpy 1.23.5、opencv-python 4.5.3.56、Pillow 10.4.0 |
| 数据集 | zarr 2.12.0、numcodecs 0.12.1 |
| 配置/权重加载 | hydra-core 1.2.0、omegaconf 2.3.1、dill 0.3.5.1 |
| 原采样器/数据检查 | scipy 1.10.1、numba 0.56.4、llvmlite 0.39.1 |
| 本地视觉编码器 | 仓库`third_party/r3m/r3m/`；已安装发行名r3m 0.0.0 |

保持这些配套版本。当前Numba元数据要求NumPy<1.24，不能单独把NumPy升级到2.x；当前diffusers与huggingface-hub也应按已验证组合保留。不要因为版本较旧就直接升级后宣称原策略行为一致。

## 在目标机新建环境

以下是提供给目标机的操作命令，本次未执行安装。在包含`deploy/`的项目根目录运行：

```bash
conda env create -f deploy/environment-idp3.yml
conda activate idp3-deploy
python -m pip check

python -c 'import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__); print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

该YAML固定主要版本，但没有锁定所有传递依赖、下载索引和Torch wheel构建，也不是已完成干净重建验证的lockfile。若目标机解析失败，保存具体错误，在独立环境处理兼容性，不改当前能工作的环境。

如要使用与开发机一致的CUDA构建，应准备对应Python3.8/Linux架构的Torch、Torchvision及配套CUDA运行库wheel。原机部分wheel在`/tmp/idp3_wheels/`，但它是临时目录且**不代表包含全部依赖**；迁移前核对清单，不能仅复制这一个目录就假定安装完整。安装后以`torch.__version__`、`torch.version.cuda`和真实推理检查为准。

完全离线重建时，先在与目标机Python/架构相匹配的环境准备完整wheelhouse，至少包含`requirements-offline.txt`及全部依赖；Conda的Python基础环境也要事先准备好。已有新建基础环境后可运行：

```bash
# 示例路径替换为已准备、完整的本地wheel目录
python -m pip install --no-index --find-links /path/to/wheelhouse \
  -r deploy/requirements-offline.txt
python -m pip check
```

如果选择整体打包现有环境，可在具备`conda-pack`的工具环境执行以下示例；当前idp3没有安装conda-pack，本次未生成压缩包：

```bash
# 开发机：输出放在项目目录，另外仍需复制项目源码、数据和正式权重
conda-pack -p /home/dingzirong/miniconda3/envs/idp3 \
  -o /path/to/project/deploy/idp3-linux-x86_64.tar.gz

# 目标机：使用新的空目录解压
mkdir -p /path/to/new-idp3
tar -xzf idp3-linux-x86_64.tar.gz -C /path/to/new-idp3
/path/to/new-idp3/bin/conda-unpack
```

完整环境含本机editable源码依赖，打包可能报告editable包或路径问题，不能静默忽略后宣称完整；先核实并迁移对应源码。环境包不能代替宿主机GPU驱动，也不能直接跨x86_64/ARM架构使用。机器人若是ARM/Jetson，应现场另行确认Torch wheel和依赖兼容性。

## 必须一起迁移的代码与权重

除了Python包，还需要：

- `deploy/`和`Improved-3D-Diffusion-Policy/diffusion_policy_3d/`。
- 本仓库修改过的`third_party/r3m/r3m/`，不能随便换另一个r3m包。
- 显式选择的正式checkpoint，以及离线验收所用真实数据。
- 只在`deploy.verify`原生pretrained加载对照时需要已有的`~/.r3m/r3m_18/config.yaml`和`model.pt`；普通部署加载器不需要下载预训练权重。

部署加载器会根据`training_root`把本地策略源码和R3M源码加入搜索路径，不要求对整个训练仓库执行`pip install -e`。在迁移后的项目根目录用`python -m deploy.replay`运行，不从错误目录直接运行脚本。

## 真机依赖需要另行补齐

本机idp3的模块查找结果：`unitree_sdk2py`、`cyclonedds`、`teleimager`、`logging_mp`、`pinocchio`均不存在。现有离线验证不依赖这些模块，不能把当前环境说成已经装好完整真机驱动。

| 后续功能 | 要补齐/核对的内容 |
| --- | --- |
| Unitree DDS读写 | 目标设备匹配的Unitree SDK及CycloneDDS；现有参考SDK的setup.py声明unitree_sdk2py 1.0.1、cyclonedds==0.10.2，实际采用版本由目标机确认 |
| 头部图像客户端 | 参考源码teleimager 1.6.0，依赖logging_mp、OpenCV、NumPy、PyYAML、pyzmq；现有idp3已装pyzmq 27.1.0和PyYAML 6.0.3，但未装客户端本身 |
| 重力前馈 | Pinocchio及与设备一致的动力学模型、惯性和关节/姿态配置；当前没有已验证版本，不填猜测版本 |

这些参考源码原位于相邻`data_collection/third_party/`。先由机器人主机的Coding Agent核对SDK、系统动态库和设备版本，再安装；不要直接把整套遥操requirements灌进idp3。策略不需要VisionPro、遥操重定向、StarVLA、语言模型或仿真环境。

注意：参考teleimager的`pyproject.toml`声明构建工具`setuptools>=61.0`，当前运行环境的setuptools为59.5.0。目标机安装应准备符合要求的隔离构建环境/预构建wheel；不应为了安装它直接在当前环境盲目使用`--no-build-isolation`或升级整套依赖。SDK安装完成也不授权启动控制器或发送命令，按[next_goal.md](next_goal.md)继续分阶段接入。

如果还需要继续RGB训练，原workspace还会导入WandB、pandas和tqdm；当前环境已装wandb 0.24.2、pandas 2.0.3、tqdm 4.70.1。这些不属于部署运行的必需包，本YAML不承诺覆盖原项目全部3D/仿真训练分支。

## 迁移后验收

```bash
# 在迁移后的项目根目录，先检查配置里的权重和数据路径
python -m deploy.replay --help
python -m unittest discover -s deploy/tests -v
python -m deploy.replay --config deploy/configs/right_pick.json \
  --output deploy/reports/environment_acceptance --max-frames 8

# 仅在GPU确实可用时；输出目录同样必须不存在
python -m deploy.replay --config deploy/configs/right_pick.json --device cuda:0 \
  --output deploy/reports/environment_acceptance_gpu --max-frames 8
```

应确认EMA/normalizer恢复成功、输出15×28且有限、真实数据回放可运行，再记录目标机推理延迟。`pip check`或import成功只能证明部分依赖状态，不能代替正式权重推理、CUDA执行及硬件联调验收。

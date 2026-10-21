# 第 1 步：可追溯的训练数据

只实现 goal.md 的第 1 步，不修改 Dataset、训练配置或模型。
转换器位于项目根目录相邻的 ../tsingtec_raw_data/tools/convert_unitree_to_zarr.py；
核对脚本为本目录的 check_unitree_zarr.py。

## 28 维如何保证顺序

state 和 action 共用一个 flatten_qpos，显式遍历
("left_arm", "right_arm", "left_ee", "right_ee")，每组取原始 qpos 的 7 个数，再拼接：

| Python 切片（从 0 开始、右端不含） | 内容 |
| --- | --- |
| [0:7] | left_arm.qpos[0:7] |
| [7:14] | right_arm.qpos[0:7] |
| [14:21] | left_ee.qpos[0:7] |
| [21:28] | right_ee.qpos[0:7] |

不遍历 JSON 字典来决定顺序。原数据 info.joint_names 的键顺序实际是
left_arm, left_ee, right_arm, right_ee, body，直接照它拼接会把右臂与左手放反。
body 为空，不参与拼接。每组内部按原数组下标逐项保留，左右手符号不变，
action 的 ±0.001 不清零；唯一数值变化是转为 float32。

states 是实际角度，actions 是绝对目标角，单位 rad。两者取同一 JSON 帧，
不因执行延迟而平移。原始关节名字为空，所以这里只保证“分组 + 组内原下标”，
不推断真实关节名、不做电机映射。28 个 group[index] 标签与切片写入 Zarr 属性和转换报告。

## 转换流程

1. 只读取 dex3_pick/right/episode_*，按 episode 数字编号排序（保留编号缺口）。
2. 每次加载一条 JSON，按 data 原顺序处理；核对连续 idx，不按时间戳重排。
3. 每帧只解码 colors.color_1；OpenCV BGR→RGB 后用 INTER_AREA 将完整视野缩放至 224×224，
   不裁剪、不归一化，保留 uint8、NHWC、0–255。
4. 一次只缓存一条 episode 的缩放图像、state、action，随后追加到 Zarr。
   图像 chunks 为 (16,224,224,3)，关节为 (256,28)，均使用 Blosc/Zstd 压缩。
5. 写入该条后追加累计 meta/episode_ends，其值是右端不含的全局结束索引。
6. 根据单调时钟记录超过 100 ms 的相邻帧间隔，只报告、不删除、不插值。
7. 核对长度和 episode 边界，保存 conversion_report.json，最后标记 conversion_complete=true。

必要检查仅包括源文件能读取、图像能解码、每组恰好 7 个有限数、连续帧号、非空 episode 和累计边界。
输出已存在时直接拒绝覆盖，不提供自动修复、自动跳帧、恢复或额外格式兼容层。
异常时保留未完成输出，完成标记仍为 false，排查后自行选择新的输出路径。

## 如何追溯和核对

报告记录源目录、转换器 SHA256、依赖版本、每条 JSON 的 SHA256、episode 编号、帧数和全局 [start,end)。
因此全局帧 k 对应该 episode JSON 的 data[k-start]，图像路径由该帧 colors.color_1 定位。

核对脚本独立按四段切片构造预期值（不调用转换器的拼接函数），
逐条比较全部 state/action 和边界；在首、中、尾 episode 的首、中、尾帧
独立重读右眼 JPEG，并以 BGR 最后一维反转为 RGB 后缩放，逐像素比较。
另外验证全部图像可按块读回、报告 JSON 哈希与当前源文件一致。
结果只能说明转换正确，不代表训练或部署已验证。

## 运行

从项目根目录执行，使用已安装依赖的 idp3 环境：

    /home/dingzirong/miniconda3/envs/idp3/bin/python \
      ../tsingtec_raw_data/tools/convert_unitree_to_zarr.py \
      --raw-dir ../tsingtec_raw_data/dex3_pick/right \
      --output data/dex3_pick_right_rgb.zarr

    /home/dingzirong/miniconda3/envs/idp3/bin/python tools/check_unitree_zarr.py

## 实际转换结果

2026-10-21 已完成：

- 输出：data/dex3_pick_right_rgb.zarr，约 707 MiB。
- img：(10344,224,224,3)，uint8；state/action 均为 (10344,28)，float32。
- episode_ends：(60,)，int64；严格递增，最后一项 10344。
- 全部 10,344 帧的 state/action 与源 JSON 转 float32 后逐项完全一致，所有 JSON 哈希一致。
- episode_0001、episode_0034、episode_0070 各首/中/尾帧，共 9 张 RGB 图逐像素一致；全部图像块可读。
- 原 action 中绝对值为 float32(0.001) 的 30,770 个数全部保留。
- 保留并报告 episode_0004/idx=123 的 110.013973 ms 和 episode_0019/idx=114 的 127.181594 ms 间隔。
- 现有 ReplayBuffer 可直接打开，读得 60 episodes、10,344 steps；转换器 SHA256 与报告一致。

完整 episode 清单和切片等元信息见 data/dex3_pick_right_rgb.zarr/conversion_report.json。
数据目录被原有 .gitignore 忽略；转换器位于项目外的指定原始数据工具目录，均不会出现在本项目 git diff 中。

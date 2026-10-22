# 离线验证实测报告（2026-10-22）

正式训练 checkpoint + 真实训练演示验证通过；未连接硬件、未发布运动命令。实际命令与工作目录保存在各个 JSON 报告的 `command` / `cwd`，选用权重 SHA-256、有效配置和随机种子一并保存。输出均在 deploy/reports，与正式权重分开。

## 正式产物与恢复

四任务的日志最后均为 epoch=300；latest 和 epoch300 文件实际存在，见 [training_inventory.json](reports/training_inventory.json)。latest 保存的是下一轮计数 epoch=301，不意味着额外训练了一轮。下表为末轮 train loss；`test_mean_score` 是负训练loss，不是测试任务成功率。

| 任务输出目录 | epoch300 train loss | latest 字节数 |
| --- | ---: | ---: |
| 2026.10.21-16.25.46_gpu4_seed42 | 0.001691529 | 1435525464 |
| 2026.10.21-16.36.38_right_place_gpu4_seed42 | 0.001989303 | 1435525464 |
| 2026.10.21-16.36.38_right_to_left_handover_low_gpu5_seed42 | 0.001723917 | 1435525528 |
| 2026.10.21-16.36.38_stack_gpu5_seed42 | 0.001801163 | 1435525400 |

四任务均严格恢复 EMA（289个参数/buffer/normalizer张量），输出15×28切片与16×28完整预测，所有数值有限。训练状态、optimizer和WandB未启动。right_pick normalizer完整保存于 [result.json](reports/right_pick_verification_final/result.json)，其action输入min/max与全部真实数据逐项相同，未重新估计；state/image为identity。

## right_pick 完整契约检查

- 全部60条、10,344帧raw state/action转float32后与Zarr逐项一致；全部源JSON与转换报告SHA-256一致。60条都没有关节名和hand_coordinates元数据，硬件零位不能离线确定。
- 每episode首/中/尾共180张RGB：独立调用原转换器，与部署raw处理及Zarr逐像素一致。没有宣称本轮重新逐像素检查全部图像。
- 全部10,404个原采样器窗口都在episode内；用原SequenceSampler和Dataset数据转换核对两条episode首/第二/尾窗口共6项，确认旧→新顺序、首帧重复、末帧padding与边界复位。
- 保留原RandomCrop/RandomRotation/ColorJitter及DDIM噪声。相同输入、Python/NumPy/Torch随机状态下，与原生pretrained R3M加载并恢复同一EMA后的原predict_action比较：完整预测和切片最大差均为0。继续随机流时同输入输出变化最大 0.05033648 rad，未逐推理重置种子。
- 真实checkpoint schema的缺失normalizer和错误/额外网络张量拒绝检查通过；不存在随机权重fallback。
- 原路径对照需要本机已有R3M cache；部署新加载器本身不需要该cache，也不会下载。对照配置、原cache和正式权重哈希在报告内。

单独对齐完整15步：episode索引0、最新观测帧40，对照演示action[40:55]，MAE 0.00333944 rad，RMSE 0.00766936 rad；最大相邻预测跳变 0.14149922 rad。

| 部位 | 15步对齐 MAE rad |
| --- | ---: |
| left_arm | 0.00088425 |
| right_arm | 0.00297367 |
| left_ee | 0.00084732 |
| right_ee | 0.00865253 |

## 四任务真实回放

right_pick选episode索引0/1/59各32帧；其余三任务各选0/59两条episode的前12帧。E=4；raw路径另用right_pick第0条前8帧、E=1。下面误差比较队列在录制时间选中的目标与同帧演示action，并非所有episode的全量评测。反馈独立来自录制，未用预测伪造反馈。

| 回放 | 推理次数 | 记录目标数 | MAE rad | RMSE rad | 平均推理 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [right_pick](reports/right_pick_replay/summary.json) | 22 | 85 | 0.00859630 | 0.02176537 | 161.00 | 163.37 |
| [right_place](reports/right_place_replay/summary.json) | 6 | 19 | 0.00525681 | 0.01197593 | 172.97 | 191.45 |
| [stack](reports/stack_replay/summary.json) | 6 | 20 | 0.00996826 | 0.02630855 | 180.67 | 187.41 |
| [right_to_left_handover_low](reports/right_to_left_handover_low_replay/summary.json) | 6 | 19 | 0.00283692 | 0.00419555 | 167.13 | 173.85 |
| [right_pick_raw](reports/right_pick_raw_replay/summary.json) | 8 | 8 | 0.00815777 | 0.01589517 | 181.59 | 196.69 |

| 回放 | 左臂 MAE | 右臂 MAE | 左手 MAE | 右手 MAE | 最大连续目标跳变 rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| right_pick | 0.00354194 | 0.02062792 | 0.00305667 | 0.00715865 | 0.45492178 |
| right_place | 0.00230741 | 0.01179353 | 0.00334614 | 0.00358015 | 0.13442394 |
| stack | 0.00175818 | 0.00955888 | 0.00221531 | 0.02634068 | 0.13351148 |
| right_to_left_handover_low | 0.00243260 | 0.00472038 | 0.00204814 | 0.00214659 | 0.03241330 |
| right_pick_raw | 0.00396140 | 0.02116920 | 0.00225680 | 0.00524367 | 0.07490034 |

每条JSONL还保存实际状态、原预测、索引和时间；没有真实发送目标。回放按录制虚拟时间推进，不把推理耗时计入虚拟执行；没有15个目标无间隔发往机器人。即使某帧无到期目标，也保留观测记录。队列末尾未到期目标在episode结束清除，不跨episode。

## 限位与时间观察

按源码候选顺序读取给定G1-D URDF，全部28个关节名存在。right_pick的289,632个state分量中0个越界，action分量中38,791个越界，全部位于手部。数据统计范围、URDF位置/速度限制、逐关节越界计数、硬件未确认null值见 [limits_comparison.json](reports/right_pick_verification_final/limits_comparison.json)。没有将演示动作范围或URDF自动用作已验证硬件边界，也没有通过裁剪掩盖差异。

保留两处录制间隔：episode_0004第123帧约110.014ms、episode_0019第114帧约127.182ms。原时间戳为录制器add_item时间，不是传感器源时间，无法据此证明图像/反馈精确同步。

可用设备为Intel Xeon Platinum 8468，CPU四线程，PyTorch2.1.0+cu121。最初CUDA探测为False / 0设备且NVML不可初始化，未使用提权或其他方式绕过限制；没有GPU性能结果。四任务推理平均约161–181ms，包含增强和DDIM，计时范围为predict_action（不含加载/图片读取/输入拷贝/记录）。部分回放进程并发，耗时是本环境样本而非隔离性能基准。

名义30Hz目标间隔33.33ms，不等于每33.33ms都能重新推理。当前CPU下默认E=4的目标窗口在推理结束时已过期，后续Session会丢弃这些目标而不是连续补发；需要目标主机实测后调整E、设备和超时。不能据此宣称目标主机达到实时要求。

## 执行逻辑与范围

9项关键单元测试全部通过，见 [unit_tests.log](reports/unit_tests.log)：历史/episode、左右目颜色、队列间隔/暂停/停止/过期、位置与速度约束交集、非有限与失效反馈、记录反馈独立、显式仿射映射和力矩provider、无发布手消息、推理后反馈复查与异常关闭。测试参数为明示测试值，不是硬件标定。

编译检查通过；import及两个CLI的--help检查通过，见 [import_and_help_check.log](reports/import_and_help_check.log)，无SDK、相机、遥操或WandB导入/构造。源码审查未发现本目录离线入口启动硬件。

上述检查使用训练集演示；误差与回放不能证明模型充分使用图像、泛化能力、真实执行反馈、机器人动力学、真机闭环或任务成功。硬件通信transport、接收时间戳、坐标/零位/限位/增益和重力模型仍需按迁移文档完成并现场核实。

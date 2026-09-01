# MuJoCo Panda Pick-and-Place 学习仓库

这是一个从最小 MuJoCo/Gym 交互逐步学习到 3D 阻尼最小二乘逆运动学（DLS IK）的项目。场景和机器人来自 `gym-hil 0.1.14`，机械臂是 **Franka Panda**。

> 本项目不是 SO-101 的仿真模型。这里的关节结构、控制器参数和实验结果不能直接部署到 SO-101 真机。

## 四个学习项目

| 阶段 | 项目 | 学习重点 |
|---|---|---|
| 1 | [ACT](https://github.com/li-clone/so101-lerobot-project) | Transformer Action Chunking，建立单指令双视角基线 |
| 2 | [Diffusion Policy](https://github.com/li-clone/so101-lerobot-diffusion-policy) | 迭代去噪动作生成与闭环恢复行为 |
| 3 | [SmolVLA](https://github.com/li-clone/so101-lerobot-smolvla) | 语言条件、多目标区域与布局互换 |
| 4 | **MuJoCo Pick-and-Place（本仓库）** | MuJoCo Jacobian、3D DLS IK、关节 PD 与内置 6D OSC 对比 |

导航体现学习演进，不是算法排行榜。前三个项目是 SO-101 真机项目，第 4 个是 Franka Panda 仿真项目；四者的数据、任务难度、评测协议和 loss 定义不同，结果不能直接横向比较。

## 1. 安装环境

```bash
cd /home/kerong/mujoco-panda-pick-place-learning
conda env create -f environment.yml
conda activate mujoco-panda-learning
```

如果已经有之前创建的 `lerobot-mujoco` 环境，也可以直接使用：

```bash
conda activate lerobot-mujoco
```

## 2. 八课学习路线

建议按顺序运行，每一步只关注一个新概念。

| 课程 | 脚本 | 学习重点 |
|---|---|---|
| 1 | `lessons/lesson_01_smoke_test.py` | `reset()`、`step()`、动作空间和观测 |
| 2 | `lessons/lesson_02_viewer.py` | MuJoCo viewer、旋转、平移和缩放视角 |
| 3 | `lessons/lesson_03_move_x.py` | TCP 沿世界坐标 +X 移动 |
| 4 | `lessons/lesson_04_move_z_observe.py` | +Z 移动和 18 维机器人状态 |
| 5 | `lessons/lesson_05_gripper.py` | 夹爪控制值与真实开合方向 |
| 6 | `lessons/lesson_06_hover_above_cube.py` | 读取方块真值并进行闭环位置控制 |
| 7 | `lessons/lesson_07_pick_place_demo.py` | 完整规则状态机 Pick-and-Place |
| 8 | `benchmark/ik_compare.py` | 自定义 3D DLS + PD 与内置 6D OSC 对比 |

无窗口脚本使用 EGL：

```bash
MUJOCO_GL=egl python lessons/lesson_01_smoke_test.py
MUJOCO_GL=egl python run.py demo
```

需要实时窗口的课程不要设置 `MUJOCO_GL=egl`：

```bash
python lessons/lesson_02_viewer.py
python run.py demo --viewer
```

## 3. 三个主要入口

规则控制器完成一次任务并保存前视、腕部相机视频：

```bash
MUJOCO_GL=egl python run.py demo
```

运行一个 DLS episode：

```bash
MUJOCO_GL=egl python run.py dls --seed-start 7
```

实时观察并保存 DLS 双相机视频：

```bash
python run.py dls --seed-start 7 --viewer --save-video
```

运行默认的 20 seeds × 2 controllers 配对评测：

```bash
MUJOCO_GL=egl python run.py compare
```

输出写入 `outputs/`，该目录不会提交到 Git。CLI 仍可继续传递底层参数，例如 `--output-dir`、`--damping`、`--max-joint-step` 和 `--joint-limit-margin`。

## 4. 控制器与任务结构

共享状态机为：

```text
OPEN → APPROACH → DESCEND → GRASP → LIFT
     → TRANSPORT → PLACE → RELEASE → RETREAT
```

内置基线通过环境的 6D operational-space controller（6D OSC）执行笛卡尔位移。自定义控制路径从 MuJoCo 获取 TCP 平移 Jacobian，求解位置 3D DLS IK，再通过 500 Hz 关节 PD 力矩内环驱动物理模型。详细说明见 [控制器笔记](docs/controller_notes.md)，Gym/MuJoCo 和 sim-to-real 概念见 [学习笔记](docs/learning_notes.md)。

## 5. 已验证结果

固定方块随机范围为 `x∈[0.30,0.42] m`、`y∈[-0.15,0.15] m`，目标区固定。MuJoCo 3.8.1 与 gym-hil 0.1.14 下的 20-seed 配对结果为：

| 控制器 | 成功率 | 最终目标距离 mean / P95 | waypoint 误差 mean / P95 | 平均控制步数 |
|---|---:|---:|---:|---:|
| 自定义 3D DLS + PD | 20/20（100%） | 22.57 / 27.19 mm | 3.40 / 5.82 mm | 181.55 |
| gym-hil 内置 6D OSC | 20/20（100%） | 20.70 / 21.64 mm | 14.13 / 17.06 mm | 183.65 |

原始明细、汇总和 seed 7 示例视频保存在 `results/reference/`。这些结果只描述当前场景和参数，不代表真机性能，也不能据此断言某个控制器普遍更优。

## 6. 自动化测试

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -p 'test_*.py' -v
```

测试覆盖 Jacobian/DLS 数值、近奇异稳定性、步长和限位、力矩裁剪、无碰撞位置收敛、固定 seed 抓放和配对结果序列化。

## 7. 常见问题

- **EGL 初始化失败**：有桌面时去掉 `MUJOCO_GL=egl`；无桌面服务器需要可用的 EGL 驱动。
- **Wayland window position 警告**：通常只表示 Wayland 不允许程序指定窗口位置，不影响仿真。
- **找不到项目模块**：确认当前目录是仓库根目录，并优先使用 `python -m ...` 或 `python run.py ...`。
- **没有生成视频**：对比脚本默认不录像，需要添加 `--save-video`；规则 demo 默认录像。
- **真机迁移**：先更换为目标机械臂模型、动作定义和控制接口，再进行 sim-to-sim 与真机安全验证，不能直接复制 Panda 力矩控制参数。

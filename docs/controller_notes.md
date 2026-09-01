# 3D DLS IK 与控制器笔记

## MuJoCo Jacobian

`controllers/dls_ik.py` 调用 `mj_jacSite` 获得指定 TCP site 相对于所有自由度的 Jacobian，并通过 `jnt_dofadr` 选出 Panda 七个关节列。平移部分最终为 `3×7` 矩阵：

```text
Jp = ∂p_tcp / ∂q
```

MuJoCo 同时负责正向运动学，因此项目不手写 Panda FK。

## 阻尼最小二乘

位置主任务使用：

```text
Δq = Jpᵀ (Jp Jpᵀ + λ²I)⁻¹ (p_target - p_tcp)
```

默认 `λ=0.05`。阻尼让接近奇异构型时的线性系统仍保持良态；每次 `Δq` 的 L2 范数限制为 `0.05 rad`。参考关节还会裁剪到模型限位以内，并保留 `0.02 rad` 安全裕量。

位置任务只有三维，而 Panda 有七个关节。实现使用弱 Jacobian 零空间 nominal-posture 正则，从冗余解中选择较稳定、夹爪朝下的构型。它不计算姿态误差，因此仍然是位置 3D IK，而不是完整 6D 姿态 IK。

## PD 力矩执行

DLS 在 scratch `MjData` 中迭代产生关节参考，不直接改写物理仿真的 `qpos`。真实执行使用：

```text
τ = Kp(qref - q) - Kd·dq + qfrc_bias
Kd = 2√Kp
Kp = [180, 180, 150, 120, 60, 45, 30]
```

`physics_dt=0.002 s`，所以内环以 500 Hz 调用 `mj_step`；外层 `control_dt=0.05 s`。最终力矩按每个 actuator 的 `ctrlrange` 裁剪，并记录裁剪子步数。

初始计划的较低 Kp 在夹持 0.1 kg 方块时产生约 8.3 mm 稳态误差，超过 6 mm waypoint 阈值。当前增益为该组增益的 1.5 倍，并经过现有任务测试，但不应直接用于真机。

## 与内置 6D OSC 的公平对比

两条控制路径使用独立环境，但同一 seed 的方块初始状态完全一致，并共享状态机、waypoint、6 mm 收敛阈值、每 waypoint 最多 80 步、settling 步骤和 4 cm 成功判据。

- `builtin`：将笛卡尔动作交给 gym-hil 内置 6D operational-space controller。
- `dls`：使用自定义位置 DLS、关节 PD 和相同夹爪 setpoint。

主要指标包括成功率、最终方块到目标距离、waypoint 误差、控制步数、关节限位裁剪、力矩裁剪和失败原因。waypoint 指标记录 settling 后的误差，所以个别值可能大于首次进入的 6 mm 阈值。

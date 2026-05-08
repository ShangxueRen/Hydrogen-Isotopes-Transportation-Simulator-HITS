# 氢同位素输运模拟程序HITs

## 1. Overview

本项目提供了一套用于模拟金属体系中氢同位素输运行为的数值程序，基于氢同位素在材料中的扩散–俘获物理模型构建。仓库包含两套完整的模拟代码：氢同位素输运模拟程序（HITs）以及考虑氦效应的氢同位素输运模拟程序（He-HITs），分别用于描述常规扩散–俘获过程以及氦泡和缺陷耦合的结构对输运行为的影响。

程序采用一维空间离散模型，支持非均匀网格划分以及温度依赖扩散系数的计算，能够描述复杂材料体系中多种类型缺陷的空间分布及其动力学过程。在此基础上，He-HITs进一步发展了扩散系数随空间变化的数值处理方法，可有效描述由氦泡结构引起的扩散阻滞效应。

在数值方法上，HITs与He-HITs均基于变步长有限差分格式构建，并采用隐式时间推进与牛顿迭代方法求解非线性耦合方程组，从而在保证计算稳定性的同时提高计算效率，适用于扩散系数及反应项强烈依赖温度与浓度的复杂体系。

该程序支持用户自定义温度历史（如线性升温过程）、缺陷分布函数、边界条件及数值求解参数，并可输出包括浓度分布、表面通量及TDS谱在内的多种物理量。通过与实验TDS数据的对比分析，可用于反演材料中缺陷类型及其关键参数。

本代码主要面向核聚变材料（如钨等等离子体面材料）中氢同位素输运与滞留行为的研究，同时也可扩展应用于其他涉及扩散–俘获过程或扩散系数具有空间非均匀性的材料体系。

This repository provides numerical simulation codes for hydrogen isotope transport in metallic systems, based on the diffusion–trapping framework. Two complete simulation codes are included: the Hydrogen Isotope Transport Simulator (HITs) and its extended version with helium effects (He-HITs), which are designed to describe conventional diffusion–trapping processes and helium-induced transport modification, respectively.

The codes are based on a one-dimensional spatial discretization scheme, supporting non-uniform grid construction and temperature-dependent diffusion coefficients. This enables the simulation of complex material systems with multiple types of defects and their associated kinetic behaviors. In particular, the He-HITs code introduces a numerical framework for spatially varying diffusion coefficients, allowing the description of diffusion retardation effects induced by helium bubble structures. 

From a numerical perspective, both HITs and He-HITs are constructed using a finite difference scheme with non-uniform grid spacing. The governing nonlinear coupled equations are solved using an implicit time integration method combined with Newton iteration, ensuring numerical stability while maintaining high computational efficiency. The framework is particularly suitable for systems where diffusion coefficients and reaction terms strongly depend on temperature and concentration.

The codes allow flexible user-defined configurations, including temperature history (e.g., linear heating), defect distribution functions, boundary conditions, and solver parameters. The outputs include concentration profiles, surface flux, and simulated TDS spectra. By comparing simulation results with experimental TDS data, the framework can be used to infer defect types and their associated parameters in materials.

These codes are primarily developed for the study of hydrogen isotope transport and retention in fusion-relevant materials, such as tungsten plasma-facing components. They can also be extended to other material systems involving diffusion–trapping processes or spatially heterogeneous diffusion behavior.

---

## 2. Physical Model

The hydrogen transport is described by a diffusion–trapping system:

$$\frac{\partial C}{\partial t} = \nabla \cdot (D(T)\nabla C) - \sum_i \[k_{t,i} C (\eta_i - C_{t,i})-k_{d,i} C_{t,i}\] + S$$

$$\frac{\partial C_{t,i}}{\partial t} = k_{t,i} C (\eta_i - C_{t,i})-k_{d,i} C_{t,i}$$

Where:

- $C$: mobile hydrogen concentration  
- $C_{t,i}$: trapped hydrogen concentration for trap i  
- $D(T)$: temperature-dependent diffusion coefficient  
- $\eta$: trap density  
- $k_{t,i}$, $k_{d,i}$: trapping and detrapping rate constants

The TDS signal is obtained from the surface flux:

$$
J = -D \frac{\partial C}{\partial x}
$$

---

## 3. Input Configuration
本节介绍 HITs 与 He-HITs 程序中的主要输入参数。程序的输入全部以 Python 字典形式组织，用户可通过修改不同配置模块，实现对空间网格、温度历史、材料体系、缺陷、结果可视化及保存数据的控制。 

整体而言，输入参数可分为以下七个部分： 
1. 一维空间格子
2. 时间与温度历史
3. 扩散系数
4. 缺陷与缺陷绘图
5. 注入参数
6. 结果可视化参数
7. 数据保存参数

### 3.1 One-Dimensional Spatial Grid
HITs与He-HITs均采用有限差分方法计算一维条件下的氢同位素输运行为。模拟中空间坐标表示样品从表面向体内的深度方向。因此，在正式计算前，需要首先对材料体系进行空间离散，建立用于数值求解的一维空间格子。

程序支持两种形式的空间格子：一种是在整个体相中步长保持不变的均匀格子，另一种是步长随空间位置变化的非均匀格子。空间格子的选择会直接影响计算精度和计算效率。对于HITs与He-HITs采用的有限差分方法，在空间离散采用二阶差分格式、时间推进采用一阶隐式格式的情况下，局部截断误差通常可表示为：

$$O(\delta x^2 + \delta t)$$

其中，$\delta x$为空间步长，$\delta t$为时间步长。因此，较小的空间步长通常可以提高计算精度，但也会增加方程组规模和计算时间。

对于He-HITs，由于扩散系数可以随空间位置发生变化，因此在扩散系数变化剧烈的区域，例如氦泡层边界附近，建议采用更细的空间网格，以更准确地描述空间变扩散系数对氢同位素输运行为的影响。对于非均匀格子，相邻网格步长不宜变化过大，否则可能降低局部差分近似的精度。

#### 3.1.1 Uniform Grid

在程序中，空间格子的输入参数通过字典 `GridData` 定义。对于均匀格子，所需的基本参数如下：

```python
GridData = {
    'Total Depth': 1.5e-3,   # [m]
    'Sec1_t': False,         # True or False
    'Sec2_t': False,         # True or False
    'maxstep': 0.5           # [um]
}
```

其中，`Sec1_t` 与 `Sec2_t` 用于控制是否启用分段变步长网格。当二者均设置为 `False` 时，程序会将输入识别为均匀格子设置。

各参数的含义如下：

- `Total Depth`：模拟体系的总厚度，单位为 m
- `maxstep`：空间步长，单位为 μm
- `Sec1_t`：是否启用第一段变步长区域
- `Sec2_t`：是否启用第二段变步长区域

需要注意的是，程序内部对空间步长进行了整除修正处理。如果输入的总厚度不是设定步长的整数倍，程序会自动对实际使用的步长进行适当调整，使总厚度能够被空间网格完整划分。因此，`maxstep` 可理解为期望的空间步长，而最终计算中采用的实际步长可能会与该输入值存在轻微差异。

#### 3.1.2 Non-uniform Grid

当 `Sec1_t` 与 `Sec2_t` 中至少有一个被设置为 `True` 时，程序会将输入识别为变步长格子。变步长格子用于在特定区域采用更小的空间步长，以提高近表面区域、缺陷富集区域或扩散系数剧烈变化区域的计算精度。

其输入参数示例如下：

```python
GridData = {
    'Total Depth': 1.5e-3,   # [m]
    'Sec1_t': True,          # True or False
    'Sec2_t': False,         # True or False
    'TransWidth': 100,       # [um]
    'Sec1': 5,               # [um]
    'minstep1': 0.002,       # [um]
    'Sec2': 40,              # [um]
    'minstep2': 0.5,         # [um]
    'maxstep': 0.5           # [um]
}
```
其中，`Sec1_t` 与 `Sec2_t` 用于控制是否启用两侧的变步长区域。当 `Sec1_t` 与 `Sec2_t` 均为 `True` 时，程序会将一维体系划分为五个区域：

```text
Surface side              Bulk region              Surface side
| Sec1 | Transition1 | Bulk | Transition2 | Sec2 |
```

`Sec1`区域对应步长`minstep1`，`Sec2`区域对应步长`minstep2`，`Bulk`区域对应步长`maxstep`，这三个区域内部的空间步长保持不变；`Transition1`中，步长会从`minstep1`线性地过度到`maxstep`，同理`Transition2`中步长会从`maxstep`线性地变化到`minstep2`。

当 `Sec1_t` 与 `Sec2_t` 中仅有一个被设置为 `True` 时，程序只会在对应一侧生成变步长区域，此时一维体系将被划分为三个区域。这里，`Sec1_t` 用于控制靠近左侧表面的变步长区域，`Sec2_t` 用于控制靠近右侧表面的变步长区域。

当 `'Sec1_t': True` 且 `'Sec2_t': False` 时，变步长区域位于左侧表面附近，体系划分方式为：

```text
Left surface side              Bulk region
| Sec1 | Transition1 | Bulk |
```

当 `'Sec1_t': False` 且 `'Sec2_t': True` 时，变步长区域位于右侧表面附近，体系划分方式为：

```text
Bulk region              Right surface side
| Bulk | Transition2 | Sec2 |
```
各参数的含义如下：

- `Total Depth` : 模拟体系的总厚度，单位为 m  
- `Sec1_t` : 是否启用第一段变步长区域
- `Sec1` : 第一段小步长区域的宽度，单位为 μm
-  `minstep1` : 第一段小步长区域中的最小步长，单位为 μm  
- `Sec2_t` : 是否启用第二段变步长区域
- `Sec2` : 第二段小步长区域的宽度，单位为 μm  
- `minstep2` : 第二段小步长区域中的最小步长，单位为 μm  
- `TransWidth` : 步长过渡区域的宽度，单位为 μm ，如果有有两段过渡区域，两段区域长度均为此处设定值   
- `maxstep` : 体相区域中的最大步长，单位为 μm  


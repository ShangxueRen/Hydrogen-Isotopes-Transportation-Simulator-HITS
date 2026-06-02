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

### 2.1 Transport of hydrogen isotopes in metals under defect effects

Defects in metallic materials, such as dislocations, vacancies, self-interstitial atoms, and grain boundaries, can generally act as trapping sites for hydrogen isotopes. When hydrogen atoms interact with these defects, hydrogen–defect complexes may form, causing the trapped hydrogen atoms to lose their ability to migrate freely through the lattice. Therefore, in defect-containing metals, the transport of hydrogen isotopes can essentially be described as a diffusion–reaction process coupled with defect trapping and detrapping.

In this program, the trapping and detrapping processes between hydrogen isotopes and defects are described using macroscopic rate equations. The macroscopic rate equation method is a continuum theory based on the mean-field approximation. Instead of explicitly tracking the microscopic motion of individual hydrogen atoms or defects, this approach uses macroscopic physical quantities, such as hydrogen concentration and defect concentration, to describe the average reaction rates between hydrogen atoms and defects.

Before presenting the governing equations, the hydrogen concentration in metals should first be classified and defined. In this work, hydrogen atoms located in the interstitial sites of the metal lattice and capable of freely migrating by atomic jumps are referred to as solute hydrogen, with their concentration denoted by $C_s$. Solute hydrogen is the main carrier of diffusive transport, and its migration is described by Fick’s law of diffusion. In contrast, hydrogen atoms trapped by defects are referred to as trapped hydrogen. The concentration of hydrogen trapped by the $i$-th type of defect is denoted by $C_{t,i}$. Physically, $C_{t,i}$ represents the concentration of hydrogen–defect complexes. This portion of hydrogen no longer participates in long-range diffusion. Its formation and dissociation are governed by trapping and detrapping processes, respectively, and are described by rate equations.

Based on these definitions, the diffusion–trapping equations for hydrogen isotopes in defect-containing metals can be written as:

$$\frac{\partial C_s}{\partial t} =\nabla \cdot (D(T)\nabla C_s)-\sum_i \left[k_{t,i} C_s (\eta_i - C_{t,i})-k_{d,i} C_{t,i}\right] + S \qquad \text{(1)}$$

$$\frac{\partial C_{t,i}}{\partial t} = k_{t,i} C_s (\eta_i - C_{t,i})-k_{d,i} C_{t,i} \qquad \text{(2)}$$

Where:

- $D(T)$ is the diffusion coefficient of hydrogen isotope atoms in the material system;
- $\eta_i$ is the spatial distribution of the concentration of the (i)-th type of defect, which represents the total concentration of available trapping sites;
- $k_{t,i}$ and $k_{d,i}$ are the trapping and detrapping rate constants for the (i)-th type of defect, respectively;
- $S$ is the source term, which can be used to describe hydrogen isotope implantation or other external input processes.

The diffusion coefficient, trapping rate constant, and detrapping rate constant are all treated as temperature-dependent kinetic parameters. The diffusion coefficient follows an Arrhenius-type expression with an additional isotope-mass correction:

$$D(T)=\frac{D_0}{\sqrt{m_{\mathrm{HI}}}}\exp\left(-\frac{E_D}{k_b T}\right)\qquad \text{(3)}$$

where:

- $D_0$ is the pre-exponential factor for hydrogen diffusion  
- $m_{\mathrm{HI}}$ is the relative atomic mass of the hydrogen isotope  
- $E_D$ is the diffusion activation energy  
- $k_b$ is the Boltzmann constant  

The term $m_{\mathrm{HI}}$ is introduced to describe the isotope effect on hydrogen isotope diffusion. For protium, deuterium, and tritium, $m_{\mathrm{HI}}$ can be taken as 1, 2, and 3, respectively. Since heavier isotopes generally have lower characteristic vibration and jump frequencies, the diffusion coefficient decreases with increasing isotope mass.

Similarly, the trapping and detrapping rate constants are expressed in Arrhenius forms:

$$k_{t,i}(T)=k_{t0,i}\exp\left(-\frac{E_{D}}{k_b T}\right) \qquad \text{(4)}$$

$$k_{d,i}(T)=k_{d0}\exp\left(-\frac{E_{D} + E_{t,i}}{k_b T}\right) \qquad \text{(5)}$$

where (k_{t0,i}) and (k_{d0,i}) are the pre-exponential factors for trapping and detrapping at the (i)-th type of defect, respectively. (E_{t,i}) is the activation energy associated with the trapping process, while (E_{d,i}) is the activation energy required for hydrogen to escape from the (i)-th type of trap. These parameters determine the temperature dependence of hydrogen exchange between mobile solute states and immobile trapped states. At low temperature, detrapping is usually suppressed and hydrogen tends to remain trapped. As temperature increases, the detrapping rate rises rapidly, enabling trapped hydrogen to be released back into the solute state and subsequently diffuse through the material.


The TDS signal is obtained from the surface flux:

$$
J = -D \frac{\partial C}{\partial x}
$$

---

## 3. Input Configuration
This section introduces the main input parameters used in the HITs and He-HITs programs. All program inputs are organized in the form of Python dictionaries. By modifying different configuration modules, users can control the spatial grid, temperature history, material system, trap settings, result visualization, and data saving options.

In general, the input parameters can be divided into the following seven parts:

1. One-dimensional spatial grid  
2. Time and temperature history  
3. Diffusion coefficient  
4. Trap and trap-plot configuration  
5. Injection parameters  
6. Result visualization parameters  
7. Data saving parameters  

### 3.1 One-Dimensional Spatial Grid
HITs and He-HITs both use the finite difference method to calculate one-dimensional hydrogen isotope transport. In the simulation, the spatial coordinate represents the depth direction from the sample surface into the bulk. Therefore, before performing the calculation, the material system must first be spatially discretized to construct a one-dimensional spatial grid for numerical solution.

The program supports two types of spatial grids: a uniform grid with constant spacing throughout the bulk material, and a non-uniform grid with spatially varying grid spacing. The choice of spatial grid directly affects both computational accuracy and efficiency. For the finite difference method adopted in HITs and He-HITs, when a second-order spatial discretization scheme and a first-order implicit time integration scheme are used, the local truncation error can generally be expressed as:

$$O(\Delta x^2 + \Delta t)$$

where $\Delta x$ is the spatial step size and $\Delta t$ is the time step. Therefore, a smaller spatial step size can generally improve computational accuracy, but it also increases the size of the equation system and the computational cost.

For He-HITs, since the diffusion coefficient can vary with spatial position, a finer spatial grid is recommended in regions where the diffusion coefficient changes significantly, such as near the boundary of the helium bubble layer. This allows the effect of spatially varying diffusion coefficients on hydrogen isotope transport to be described more accurately. For non-uniform grids, excessively large changes between adjacent grid spacings should be avoided, as they may reduce the local accuracy of the finite difference approximation.

#### 3.1.1 Uniform Grid

In the program, the input parameters for the spatial grid are defined using the dictionary `GridData`. For a uniform grid, the basic required parameters are as follows:

```python
GridData = {
    'Total Depth': 1.5e-3,   # [m]
    'Sec1_t': False,         # True or False
    'Sec2_t': False,         # True or False
    'maxstep': 0.5           # [um]
}
```

Here, `Sec1_t` and `Sec2_t` are used to control whether segmented non-uniform grid regions are enabled. When both of them are set to `False`, the program recognizes the input as a uniform grid configuration.

The meanings of the parameters are as follows:

- `Total Depth`: the total thickness of the simulated system, in m
- `maxstep`: the spatial step size, in μm
- `Sec1_t`: whether to enable the first non-uniform grid region
- `Sec2_t`: whether to enable the second non-uniform grid region

It should be noted that the program internally applies a divisibility correction to the spatial step size. If the input total thickness is not an integer multiple of the specified step size, the program will automatically adjust the actual step size used in the calculation so that the total depth can be completely divided by the spatial grid. Therefore, `maxstep` can be understood as the expected spatial step size, while the actual step size used in the final calculation may differ slightly from the input value.

#### 3.1.2 Non-uniform Grid

When at least one of `Sec1_t` and `Sec2_t` is set to `True`, the program recognizes the input as a non-uniform grid configuration. The non-uniform grid is used to apply smaller spatial step sizes in specific regions, thereby improving the computational accuracy in near-surface regions, defect-enriched regions, or regions where the diffusion coefficient changes significantly.

An example of the input parameters is given below:

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

Here, `Sec1_t` and `Sec2_t` are used to control whether the non-uniform grid regions on the two sides of the one-dimensional domain are enabled. When both `Sec1_t` and `Sec2_t` are set to `True`, the program divides the one-dimensional system into five regions:

```text
Surface side              Bulk region              Surface side
| Sec1 | Transition1 | Bulk | Transition2 | Sec2 |
```

The `Sec1` region uses the spatial step size `minstep1`, the `Sec2` region uses the spatial step size `minstep2`, and the `Bulk` region uses the spatial step size `maxstep`. Within these three regions, the spatial step size remains constant. In the `Transition1` region, the step size linearly increases from `minstep1` to `maxstep`. Similarly, in the `Transition2` region, the step size linearly decreases from `maxstep` to `minstep2`.

When only one of `Sec1_t` and `Sec2_t` is set to `True`, the program generates a non-uniform grid region only on the corresponding side, and the one-dimensional system is divided into three regions. Here, `Sec1_t` controls the non-uniform grid region near the left surface, while `Sec2_t` controls the non-uniform grid region near the right surface.

When `'Sec1_t': True` and `'Sec2_t': False`, the non-uniform grid region is located near the left surface, and the system is divided as follows:

```text
Left surface side              Bulk region
| Sec1 | Transition1 | Bulk |
```

When `'Sec1_t': False` and `'Sec2_t': True`, the non-uniform grid region is located near the right surface, and the system is divided as follows:

```text
Bulk region              Right surface side
| Bulk | Transition2 | Sec2 |
```

The meanings of the parameters are as follows:

- `Total Depth`: the total thickness of the simulated system, in m
- `Sec1_t`: whether to enable the first non-uniform grid region
- `Sec1`: the width of the first fine-grid region, in μm
- `minstep1`: the minimum step size in the first fine-grid region, in μm
- `Sec2_t`: whether to enable the second non-uniform grid region
- `Sec2`: the width of the second fine-grid region, in μm
- `minstep2`: the minimum step size in the second fine-grid region, in μm
- `TransWidth`: the width of the transition region, in μm. If two transition regions are enabled, both transition regions use this specified width
- `maxstep`: the maximum step size in the bulk region, in μm

### 3.2 Time and temperature history

Both HITs and He-HITs use a backward difference scheme for time discretization. In the current version, the program does not use a variable time-step scheme. Instead, the entire simulation is advanced using a fixed time step `dt`. Benefiting from the simplicity of the finite difference algorithm and the use of non-uniform spatial grids, the program can still maintain high computational efficiency even when a relatively small time step and a large number of time iterations are used.

The input parameters for time and temperature history are defined using the dictionary `TempDefin`, as shown below:

```python
TempDefin = {
    'Total Time': 700,              # [s]
    'dt': 0.05,                     # [s]
    'temperature function': 'TempRampFunc',
    'temp func defin': {
        'Temperature initial': 300, # [K]
        'TPD_start': 0,             # [s], time when TPD starts
        'TPD_rate': 1.0,            # [K/s]
        'TPD_end': 800              # [s], time when TPD ends
    }
}
```

Here, `Total Time` and `dt` are used to define the time discretization:

* `Total Time`: the total duration of the simulation, in s
* `dt`: the time interval for each calculation step, in s

`temperature function` and `temp func defin` are used to define the temperature history:

* `temperature function`: the name of the temperature function used in the simulation
* `temp func defin`: the parameter-definition dictionary corresponding to the selected temperature function

The temperature functions currently implemented in the program include:

* `ConstTempFunc`: generates a constant temperature array corresponding to the time list
* `TempRampFunc`: generates a temperature array with linear heating or cooling

Among them, `TempRampFunc` can handle most common simulation scenarios, such as a linear heating process in TDS simulations, a linear cooling process after gas loading, and isothermal holding after heating or cooling.

For example, when simulating the experimental condition of cooling after gas loading, the temperature-definition dictionary can be set as follows:

```python
'temp func defin': {
    'Temperature initial': 473, # [K]
    'TPD_start': 3600,          # [s]
    'TPD_rate': -0.1667,        # [K/s]
    'TPD_end': 4637             # [s]
}
```

With this setting, the program keeps the initial temperature at 473 K for the first 3600 s. It then starts linear cooling at 3600 s with a rate of `-0.1667 K/s`. After the time reaches `TPD_end`, the temperature remains approximately at the value reached at the end of the cooling process. In this example, the system cools down to around 300 K and then remains at that temperature.

The program also supports user-defined temperature functions. The definition rules for custom temperature functions can be found in the comments in Part 3. In general, a custom temperature function should satisfy the following requirements:

- The function inputs should include `timelist` and `tempdata`
- `timelist` is the time list
- `tempdata` is the variable containing the temperature-definition information, usually in the form of a dictionary
- The function should return a numpy array
- The length of the returned temperature array must be the same as the length of `timelist`

As long as the custom temperature function satisfies the above format requirements, the existing `GetTempFunc` calling method can be used to generate the corresponding temperature-history list without modifying the main program structure.

### 3.3 Diffusion Coefficient

In the program, the diffusion coefficient of hydrogen isotopes in the material is described using the Arrhenius form:

$$
D(T) = D_0 \exp\left(-\frac{E_D}{k_b T}\right)
$$

where $D(T)$ is the diffusion coefficient at temperature $T$, $D_0$ is the pre-exponential factor, $E_D$ is the diffusion energy barrier, and $k_b$ is the Boltzmann constant.

The parameters related to the diffusion coefficient are defined using the dictionary `DFactorDefin`, as shown below:

```python
DFactorDefin = {
    'D_0': 4.1e-7 / 2 ** 0.5,      # for D, [m^2/s]
    'D_E': 0.39,                   # [eV]
    'lattice constant': 316        # [pm]
}
```

The meanings of the parameters are as follows:

* `D_0`: the pre-exponential factor of the diffusion coefficient, in $\mathrm{m}^2 \mathrm{s}^{-1}$
* `D_E`: the diffusion energy barrier, in eV
* `lattice constant`: the lattice constant of the material system, in pm

It should be noted that `lattice constant` is not directly used in the calculation of the diffusion coefficient $D(T)$. Instead, it is used to generate the pre-exponential factor of the trapping rate constant for defects. Since the lattice constant and the diffusion coefficient parameters are both intrinsic physical parameters of the material system, they are defined together in the `DFactorDefin` dictionary.

In HITs, the diffusion coefficient is usually determined only by temperature and is used to describe thermally activated diffusion in a homogeneous material system. In He-HITs, the program can further introduce a spatially dependent effective diffusion coefficient to describe the retardation effect of helium bubble layers or other heterogeneous structures on hydrogen isotope diffusion.



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

## 2. Physical Model

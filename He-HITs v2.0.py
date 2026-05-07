'''
Helium Effect - Hydrogen Isotope Transport Simulator, He-HITs v2.0
Shangyin Liu 26.4.22
'''
import os
import time
import math as m
#import random as r
import numpy as np
import scipy as sp
import pandas as pd
from pprint import pformat
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

'''
==================== Part 1 ====================
             input section 输入部分
'''
GridData = {
    'Total Depth' : 1.5e-3,         #[m]  
    'Sec1_t' : True,                #True&False
    'Sec2_t' : False,               #True&False
    'TransWidth' : 100,             #[um]
    'Sec1' : 0.030,                  #[um]
    'minstep1' : 0.00005,             #[um]
    'Sec2' : 60,                    #[um]
    'minstep2' : 0.5,               #[um]
    'maxstep' : 0.35                 #[um]
    }
TempDefin = {
    'time function' : 'TDStimefunc',
    'time datas' : {
        'tds time' : 800,
        'dt' : 0.02,
        'temperature function' : 'TempRampFunc',
        'temp func defin' : {
                                'Temperature initial' : 300,
                                'TPD_start' : 0,
                                'TPD_rate' : 1.0,
                                'TPD_end' : 800
                                }
    }}
DFactorDefin = {
    'D_0' : 4.1e-7 / 2 ** 0.5,     #for D , [m^2/s]
    'D_E' : 0.39,                  #[eV]
    'lattice constant' : 316       #[pm]
    }
ImpantationData = {
    'implantation range' : 1e-9 ,     #[m]
    'impantation straggle' : 2e-9 ,   #[m]
    'implantation flux' : 1e20 ,      #[#/(m^2 s)]
    'implantation step steepness' : 1e-5
    }
TrapPlot = {
    'plot' : True,
    'multiple plots' : True, 
    'plot range' : 0.6e-6
    }
TrapData = {
    'trap1' : {'ETS' : 1.37, 
               'beta0' : 2e13, 
               'distri_fuc' : 'nodeintervaltrapprof', 
               'distri_data' : [
                   {'cleft' : 2.8e-3, 'pos' : 30e-9},
                   {'cleft' : 3.1e-3, 'pos' : 190e-9},
                   {'cleft' : 2.1e-3, 'pos' : 270e-9},
                   {'cleft' : 4.3e-4, 'pos' : 570e-9},
                   {'cleft' : 3.5e-4, 'pos' : 1600e-9},
                   {'cleft' : 1.3e-4, 'pos' : 6e-6},
                   {'cleft' : 10e-6, 'pos' : 15e-6},
                   {'cleft' : 9.0e-6, 'pos' : 70e-6}
                   ], 'trap factor plot' : False},
    'trap2' : {'ETS' : 1.50,
               'beta0' : 2e13, 
               'distri_fuc' : 'nodeintervaltrapprof',
               'distri_data' : [
                   {'cleft' : 1.1e-2, 'pos' : 5e-9},
                   {'cleft' : 1.3e-2, 'pos' : 30e-9},
                   {'cleft' : 9.0e-4, 'pos' : 70e-9}
                   ], 'trap factor plot' : False},
    'He bubble' : {'theta_He' : 0.23, 
               'f' : 0.65, 
               'distri_fuc' : 'Hedistributefunc', 
               'distri_data' : {
                   'A' : 1004783.72, 
                   'B' : -0.008534, 
                   'C' : 0.000040,
                   'conc1' : 1e-3}}}
SolvePDE = {
    'left boundary condition' : ['Diriclet', 1], 
    'right boundary condition' : False, 
    'iteration max times' : 50,
    'iteration tolerance' : 1e-11,
    'reference concentration' : 2e-9       #1% of max Concentration
    }
SaveDatas = {
    'output' : True,
    'location' : r'D:\file datas',
    'file name' : "补充数据--He-D (5e24)",
    'save option' : {
        'if save input' : True,
        'if save grid' : True,
        'if save temp' : True,
        'if save conc' : True,
        'if save c_out' : True
        }
    }
Exdatas = {
    'Get exdata' : True,
    'location' : r'D:\输运程序\20-小论文\raw datas\He-D (5e24).csv'
    }
#with open('C:/Users/dell/Desktop/输运程序/14-柱状晶项目模拟/input files/重离子-氦-氘顺序等离子体辐照/pure-D_4.21ver2.txt', 'r', encoding='utf-8') as file:
#    content = file.read()
#exec(content)
'''
==================== Part 2 ====================
           Handy Functions 常用计算函数
'''
kb = 8.618339e-5       #[eV/K]
DfW = 6.02e28          #[#/m3]
scaletrans = 1e6       #[m]->[um]
def GaussFunc(x, center, width, Xmax):
    LOG1000 = np.log(1000.0)
    THRESHOLD_FACTOR = np.sqrt(2 * LOG1000)
    d = x - center
    w_sq = width ** 2
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x)
        d_norm = np.abs(d) / width
        mask = d_norm <= THRESHOLD_FACTOR
        exponent = -0.5 * (d[mask]**2) / w_sq
        result[mask] = Xmax * np.exp(exponent)
        return result
    if abs(d) > THRESHOLD_FACTOR * width:
        return 0.0
    return Xmax * np.exp(-0.5 * (d**2) / w_sq)
def StepFunc(x_in, stepPos, leftconc, rightconc=0):
    return np.where(x_in <= stepPos, leftconc, rightconc)
'''
==================== Part 3 ====================
        Temperature Function 温度函数
定义时间与温度函数
                   对应时间函数的参数 
输入为选择时间函数 + 对应时间段的温度函数
                   对应温度函数参数
对于时间函数，选项
TDStimefunc  --> 只进行TDS升温模拟
Impltimefunc --> 只进行辐照/注入过程模拟
IRTtimefunc  --> Implantation-Rest-TDS 进行注入-脱附全流程模拟
'''
def TDStimefunc(timedata):
    '''
    input: dict
        'time datas' : {
            'tds time' : 1000,                         #TDS持续的总时间
            'dt' : 0.5,                                #在TDS阶段的时间步长
            'temperature function' : 'TempRampFunc',   #选择TDS的温度函数，通常为温度升高的线性函数
            'temp func defin' : {
                                    'Temperature initial' : 300,
                                    'TPD_start' : 0,
                                    'TPD_rate' : 1.0,
                                    'TPD_end' : 1000
                                    }
            }
    output:
        tdstime: global,np.array, tds时间（从0计时），规模[Ntt * 1]
        tdt:     global,int/float, tds时间间隔
    '''
    totaltime = timedata['tds time']
    global tdstime, tdt
    tdt = timedata['dt']
    nt = int(totaltime / tdt) + 1
    tdstime = np.linspace(0, totaltime, nt)
def Impltimefunc(timedata):
    '''
    input: dict
        'time datas' : {
            'impl time' : 10000,                       #注入过程持续的时间
            'dt' : 2.0,                                #注入过程的时间步长
            'temperature function' : 'ConstTempFunc',  #选择注入过程的温度函数，通常为恒温函数
            'temp func defin' : {'Temperature initial' : 300}
            }
    output:
        impltime: global,np.array, 注入过程时间（从0计时），规模[Nit * 1]
        idt:      global,int/float, 注入过程时间间隔
    '''
    totaltime = timedata['impl time']
    global impltime, idt
    idt = timedata['dt']
    nt = int(totaltime / idt) + 1
    impltime = np.linspace(0, totaltime, nt)
def IRTtimefunc(timedata):
    '''
    input: dict
        'time datas' : {
            'Implantation' : {
                'impl time' : 10000,                         #注入过程持续的时间
                'dt' : 2.0,                                  #注入过程的时间步长
                'temperature function' : 'ConstTempFunc',    #选择注入过程的温度函数，通常为恒温函数
                'temp func defin' : {'Temperature initial' : 500}
                }, 
            'Rest' : {
                'rest time' : 5000,                          #放置过程持续的时间
                'dt' : 2.0,                                  #放置过程的时间步长
                'temperature function' : 'TempRampFunc',     #选择放置过程的温度函数
                'temp func defin' : {
                                        'Temperature initial' : 500, 
                                        'TPD_start' : 0,
                                        'TPD_rate' : -3.33,
                                        'TPD_end' : 60
                                        }
                },
            'TDS' : {
                'tds time' : 1000,                           #TDS持续的总时间
                'dt' : 0.5,                                  #在TDS阶段的时间步长
                'temperature function' : 'TempRampFunc',     #选择TDS的温度函数
                'temp func defin' : {
                                        'Temperature initial' : 300,
                                        'TPD_start' : 0,
                                        'TPD_rate' : 1,
                                        'TPD_end' : 1000
                                        }
                }
            }
    output:
        impltime: global,np.array, 注入过程时间（从0计时），规模[Nit * 1]
        idt:      global,int/float, 注入过程时间间隔
        resttime: global,np.array, 放置过程时间（从0计时），规模[Nrt * 1]
        rdt:      global,int/float, 放置过程时间间隔
        tdstime:  global,np.array, tds时间（从0计时），规模[Ntt * 1]
        tdt:      global,int/float, tds时间间隔
    '''
    Implparameter = timedata['Implantation']
    Restparameter = timedata['Rest']
    TDSparameter = timedata['TDS']
    global impltime, resttime, tdstime, idt, rdt, tdt
    impltot = Implparameter['impl time']
    idt = Implparameter['dt']
    nt = int(impltot / idt) + 1
    impltime = np.linspace(0, impltot, nt)
    resttot = Restparameter['rest time']
    rdt = Restparameter['dt']
    nt = int(resttot / rdt) + 1
    resttime = np.linspace(0, resttot, nt)
    tdstot = TDSparameter['tds time']
    tdt = TDSparameter['dt']
    nt = int(tdstot / tdt) + 1
    tdstime = np.linspace(0, tdstot, nt)
def Gett(TimeDefData):
    time_func = eval(TimeDefData['time function'])
    timedefdata = TimeDefData['time datas']
    time_func(timedefdata)
def ConstTempFunc(time_list, tempdata):
    '''
    input:
        time_list : np.array, 时间数组
        tempdata : dict, 
        'temp func defin' : {
            'Temperature initial' : 500    #恒温函数，需要温度值，单位为[K]
            }
    output:
        temp_list: np.array, 温度数组，规格[Nt * 1]
    '''
    n = len(time_list)
    return np.full(n, tempdata['Temperature initial'])
def TempRampFunc(time_list, tempdata):
    '''
    input:
        time_list : np.array, 时间数组
        tempdata : dict
        'temp func defin' : {
            'Temperature initial' : 300,   #初始温度值，单位为[K]
            'TPD_start' : 0,               #升温/降温开始的时间点，时间点 前 为恒温
            'TPD_rate' : 1,                #升温/降温速率，单位为[K/s]
            'TPD_end' : 1000               #升温降温结束的时间点，时间点 后 为恒温
            }
    output:
        temp_list: np.array, 温度数组，规格[Nt * 1]
    '''
    t = np.asarray(time_list)
    T0 = tempdata['Temperature initial']
    t_str = tempdata['TPD_start']
    t_end = tempdata['TPD_end']
    T_rate = tempdata['TPD_rate']
    delta_t = t - t_str
    clipped_delta_t = np.clip(delta_t, 0, t_end - t_str)
    return T0 + T_rate * clipped_delta_t
def GetTemp(tempdef):
    timeopt = tempdef['time function']
    global impltemp, resttemp, tdstemp
    if timeopt == 'TDStimefunc':
        temp_func = eval(tempdef['time datas']['temperature function'])
        tempdefindata = tempdef['time datas']['temp func defin']
        tdstemp = temp_func(tdstime, tempdefindata)
    elif timeopt == 'Impltimefunc':
        temp_func = eval(tempdef['time datas']['temperature function'])
        tempdefindata = tempdef['time datas']['temp func defin']
        impltemp = temp_func(impltime, tempdefindata)
    else:
        Implparameter = tempdef['time datas']['Implantation']
        Restparameter = tempdef['time datas']['Rest']
        TDSparameter = tempdef['time datas']['TDS']
        temp_func = eval(Implparameter['temperature function'])
        tempdefindata = Implparameter['temp func defin']
        impltemp = temp_func(impltime, tempdefindata)
        temp_func = eval(Restparameter['temperature function'])
        tempdefindata = Restparameter['temp func defin']
        resttemp = temp_func(resttime, tempdefindata)
        temp_func = eval(TDSparameter['temperature function'])
        tempdefindata = TDSparameter['temp func defin']
        tdstemp = temp_func(tdstime, tempdefindata)
'''
==================== Part 4 ====================
        Diffusion Rate Function 扩散系数
扩散系数的输入统一为 时间函数 和 扩散系数相关参数
扩散系数输出同意为一维numpy数组
'''
def DifFactor(temp_list, Difffactor):
    '''
    input:
        temp_list: np.array, 温度数组
        Diffactor: dict
        DFactorDefin = {
            'D_0' : 4.1e-7 / 2 ** 0.5,     #扩散系数指前因子，单位[m^2/s]
            'D_E' : 0.39,                  #扩散能垒，单位[eV]
            'lattice constant' : 316       #金属晶格常数[pm]
            }
    output:
        D_factor_list: np.array, 扩散系数数组，规模[Nt * 1]
    '''
    D_0 = Difffactor['D_0']
    D_E = Difffactor['D_E']
    scale_factor = D_0 * scaletrans ** 2
    exponent = -D_E / (kb * temp_list)
    D_list = scale_factor * np.exp(exponent)
    return D_list
class He_Inf_Diff:
    def __init__(self, Dfactor, He_distri_dic, pHeData):
        '''
        初始化携带氦效应的扩散系数生成函数
        Parameters
        ----------
        Dfactor : dict
            扩散系数参数字典，为原始输入，基本格式：
                DFactorDefin = {
                    'D_0' : 4.1e-7 / 2 ** 0.5, #扩散系数指前因子，单位[m^2/s]
                    'D_E' : 0.39,              #扩散能垒，单位[eV]
                    'lattice constant' : 316   #金属晶格常数[pm]
                    }
        He_distri_dic : dict
            氦泡分布参数列表，为中间计算结果，扩散系数生成函数不计算氦泡分布。
                He_distri = {
                    'He bubble list' : Hebubbledistrilist, #氦泡分布列表
                    'zero mask' : zeromask,                #掩码，零浓度位置为True
                    'non zero mask' : non_zeromask         #掩码，非零浓度位置为True
                    }
        pHeData : dict
            氦泡计算的其余相关参数，为原始输入，基本格式：
                'He bubble' : {
                    'theta_He' : 0.23,     #氦泡体积分数
                    'f' : 0.2,             #氦泡因子
                    'distri_fuc' : 'constgausstrapprof', #氦泡分布的相关信息不会在函数中使用
                    'distri_data' : {'center' : 10e-9, 'height' : 30e-3, 'width' : 10e-9}
                    }

        Returns
        None.

        '''
        xi = np.asarray(He_distri_dic['He bubble list'], dtype=np.float64)
        zero_mask = np.asarray(He_distri_dic['zero mask'], dtype=bool)
        non_zero_mask = np.asarray(He_distri_dic['non zero mask'], dtype=bool)

        D0_base = float(Dfactor['D_0']) * float(scaletrans) ** 2
        E_D = float(Dfactor['D_E'])

        Vf = float(pHeData['theta_He'])
        Bf = float(pHeData['f'])

        D0_he = (1.0 - Vf)**1.33*D0_base

        #预分配每个点的前因子和能垒
        A = np.full(xi.shape, D0_he, dtype = np.float64)
        E = np.full(xi.shape, E_D, dtype = np.float64)
        if np.any(zero_mask):
            A[zero_mask] = D0_base
            E[zero_mask] = E_D

        #有氦泡位置：预计算能垒
        if np.any(non_zero_mask):
            xi_min = float(np.min(xi))
            xi_max = float(np.max(xi))
            delta_xi = xi_max - xi_min

            xi_nz = xi[non_zero_mask]

            if delta_xi > 0.0:  #判断退化
                E[non_zero_mask] = E_D*(1 + Bf*(xi_nz - xi_min)/delta_xi)
            else:
                E[non_zero_mask] = E_D

        #只保存真正需要的常量
        self._A = A
        self._minus_E_over_kb = -E/kb

        #内部工作缓冲区：避免每次调用都重新分配数组
        self._buffer = np.empty_like(A)
        self._buffer_with_bc = np.empty(len(A) + 2, dtype=np.float64)

    def __call__(self, temp):
        '''
        扩散系数列表生成，输入温度，输出对应的扩散系数空间分布。
        Parameters
        temp : float
            当前温度值，为单个浮点数.
        
        Returns
        D_list: np.array of float64, size = [Nx+2]
            输出为Nx+2长度的扩散系数列表，多余的两个扩散系数值D[0]=D[1], D[-1]=D[-2]为边界值.
        '''
        T = float(temp)
        
        np.exp(self._minus_E_over_kb / T, out=self._buffer)
        self._buffer *= self._A
        
        self._buffer_with_bc[1:-1] = self._buffer
        self._buffer_with_bc[0] = self._buffer_with_bc[1]
        self._buffer_with_bc[-1] = self._buffer_with_bc[-2]
        return self._buffer_with_bc

def impl_rate_surf(xlist, impldata):
    '''
    ImpantationData = {
        'implantation range' : 1e-9 ,     #[m]
        'impantation straggle' : 2e-9 ,   #[m]
        'implantation flux' : 1e20 ,      #[#/(m^2 s)]
        'implantation step steepness' : 1e-5
        }
    '''
    source = np.zeros(len(xlist))
    center = impldata['implantation range'] * scaletrans
    width = impldata['impantation straggle'] * scaletrans
    flx = impldata['implantation flux'] / (scaletrans ** 2)
    maxc = 1.0
    nrm, _ = sp.integrate.quad(GaussFunc, 0, np.inf, args=(center, width , maxc))
    dens = DfW / scaletrans ** 3
    for i in range(len(xlist)):
        source[i]=flx / (nrm * dens) * GaussFunc(xlist[i], center, width , maxc)
    return source
'''
==================== Part 5 ====================
               Building Grid 格子
变步长格子
在靠近表面的部分变化缓慢，在金属内部变化迅速
通过定义输入的input列表来定义变化的范围
目前使用的为
表面         |        主体         |      表面
Sec1 | Transition1 | bulk | Transition2 |Sec3
'''
def GridTrans(width, step1, step2): 
    print('step--from :' , step1 , 'to :' , step2)
    stepnum = m.ceil((step1 - step2 + 2 * width) / (step1 + step2))
    addstep = (step2 - step1) / stepnum
    Transition1 = np.zeros(stepnum)
    Transition = np.zeros(stepnum)
    for i in range(stepnum):
        Transition1[i] = step1 + (i + 1) * addstep
        Transition[i] = sum(Transition1)
    scale = width / Transition[-1]
    Transition = Transition * scale
    Transition1 = Transition1 * scale
    print('variation range :' , width , ', num :' , stepnum)
    return Transition, Transition1
def Section(width, step): 
    nstep = m.floor(width / step)
    lim = m.floor((width - step) / step)
    lstep = (width - step) / lim
    section0 = np.zeros(nstep)
    section1 = np.zeros(nstep)
    section0[0] = section1[0] = step
    for i in range(1, nstep):
        section0[i] = lstep
        section1[i] = section1[i-1] + lstep
    print('Section width :' , width)
    print('step :' , step , ', lstep :' , lstep , ', num :' , lim)
    return section1, section0
def BuildGridFunc(pGridData):
    datas = [pGridData['Total Depth'] * scaletrans , pGridData['maxstep']]
    if pGridData['Sec1_t'] == True:
        datas.append(pGridData['Sec1'])
        datas.append(pGridData['minstep1'])
        datas.append(pGridData['TransWidth'])
    if pGridData['Sec2_t'] == True:
        datas.append(pGridData['Sec2'])
        datas.append(pGridData['minstep2'])
        datas.append(pGridData['TransWidth'])
    num = len(datas)
    if num == 2:
        print('linear step function:')
        X_list , dx_list = Section(datas[0] , datas[1])
        Numx = len(X_list)
    elif num == 8:
        print('nonlinear step function:')
        maxwidth = datas[0] - 2 * datas[4] - datas[2] - datas[5]
        x1 , d1 = Section(datas[2] , datas[3])
        x2 , d2 = GridTrans(datas[4] , datas[3] , datas[1])
        x3 , d3 = Section(maxwidth , datas[1])
        x4 , d4 = GridTrans(datas[7] , datas[1] , datas[6])
        x5 , d5 = Section(datas[5] , datas[6])
        x2 += x1[-1]
        x3 += x2[-1]
        x4 += x3[-1]
        x5 += x4[-1]
        X_list = np.concatenate((x1 , x2 , x3 , x4 , x5) , axis=0)
        dx_list = np.concatenate((d1 , d2 , d3 , d4 , d5) , axis=0)
        Numx = len(X_list)
        print('total number of X :' , Numx)
    else:
        print('nonlinear step function:')
        maxwidth = datas[0] - datas[4] - datas[2]
        if pGridData['Sec1_t'] == True:
            x1 , d1 = Section(datas[2] , datas[3])
            x2 , d2 = GridTrans(datas[4] , datas[3] , datas[1])
            x3 , d3 = Section(maxwidth , datas[1])
            x2 += x1[-1]
            x3 += x2[-1]
        else:
            x1 , d1 = Section(maxwidth , datas[1])
            x2 , d2 = GridTrans(datas[4] , datas[1] , datas[3])
            x3 , d3 = Section(datas[2] , datas[3])
            x2 += x1[-1]
            x3 += x2[-1]
        X_list = np.concatenate((x1 , x2 , x3) , axis=0)
        dx_list = np.concatenate((d1 , d2 , d3) , axis=0)
        Numx = len(X_list)
        print('total number of X :' , Numx)
    return X_list , Numx , dx_list
'''
==================== Part 6 ====================
              Trapping Modle 缺陷
每种缺陷分布需要提前给定，
缺陷分布函数f(x , d)
input:
    x : Numpy.array of float64, 空间分布列表
    d : dict, 函数要求的缺陷空间分布信息
return:
    Numpy.array of float64, 缺陷浓度分布，相对浓度单位为[#/W]
缺陷部分的标准input格式（字典单元）：
'trap1' : {'ETS' : 1.4, <--- Energy_{trapped to solute} 缺陷去捕获能垒 单位为[eV]
           'beta0' : 2e13, <--- 缺陷去捕获系数的指前因子，对于氘，常见值2e13，单位为[#/s]
           'alpha0' : 8.4e12, <--- 缺陷捕获系数指前因子  可以不提供，如提供，按照提供值计算，如不提供，按照文献值计算
           'distri_fuc' : 'mstepconcindeptrapprof', <--- 缺陷分布函数名，取下面列出的几种，或自己写个新的
           'distri_data' : <--- 缺陷分布函数需要的信息，详见每种缺陷函数
               [{'cleft' : 3e-2, 'cright' : 0, 'pos' : 13e-9, 'steepness' : 1e-1}, 
               {'cleft' : 7e-4, 'cright' : 0, 'pos' : 190e-9, 'steepness' : 1e-1}], 
           'trap factor plot' : False <--- 选择是否为缺陷反应系数画图
           }
'''
def concindeptrapprof(x, data):
    '''
    阶梯函数，在给定点'position'左侧为C_{left}，右侧为C_{right}，右侧浓度可以不提供，默认为零
    input : dict
        'distri_data' : {
            'pos' : 13e-9, <---浓度突变点的位置，单位为[m]
            'cleft' : 3e-2, <---从 0 到 突变点 处的浓度值，采用相对浓度，单位#/W atom，乘以钨的数密度得到绝对浓度
            'cright' : 0 <---突变点 到 右侧边界 处的浓度值，可以不给出，默认为零
            }
    return : array of float64
    '''
    pos = data['pos'] * scaletrans
    cleft = data['cleft']
    cright = data.get('cright', 0)
    x_arr = np.asarray(x)
    return StepFunc(x_arr, pos, cleft, cright)
def constgausstrapprof(x, data):
    '''
    高斯分布函数，生成一个单个高斯峰的缺陷分布
    input : dict
        'distri_data' : {
            'center' : 10e-9, <---高斯分布的中心位置，单位为[m]
            'height' : 3e-3, <---高斯分布的峰值浓度，采用相对浓度，单位#/W atom，乘以钨的数密度得到绝对浓度
            'width' : 10e-9 <---高斯分布的半峰宽，单位为[m]
            }
    output : array of float64
    '''
    center = data['center'] * scaletrans
    height = data['height']
    width = data['width'] * scaletrans
    return GaussFunc(x, center, width, height)
def mstepconcindeptrapprof(x, data):
    '''
    复合阶梯函数分布，用于生成多段阶梯函数，模仿类似TMAP7的输入效果
    input : list <---输入为一个列表，每个元素为字典，字典内容见concindeptrapprof函数注释
        'distri_data' : [
            {'cleft' : 3e-2, 'pos' : 13e-9}, 
            {'cleft' : 7e-4, 'pos' : 190e-9}
            ]
    output : array of float64
    '''
    params = np.array([(
        item['cleft'],
        item['pos'] * scaletrans,
        item.get('cright', 0),
    ) for item in data], dtype=np.float64)
    x_arr = np.asarray(x)[:, np.newaxis]
    pos_arr = params[:, 1]
    mask = x_arr <= pos_arr
    concentrations = np.where(mask, params[:, 0], params[:, 2])
    return np.sum(concentrations, axis=1)
def constanttrapprof(x, data):
    '''
    常值函数，生成整个样品深度范围内相同的均匀缺陷
    input : dict
        'distri_data' : {
            'conc' : 2e-5 <---均匀分布缺陷的浓度 采用相对浓度，单位#/W atom，乘以钨的数密度得到绝对浓度
            }
    output : array of float64
    '''
    concentration = data['conc']
    return np.full(len(x), concentration, dtype=np.float64)
def nodeintervaltrapprof(x, data):
    params = sorted(data, key=lambda item: item['pos'])
    pos_arr = np.array([item['pos'] * scaletrans for item in params], dtype=np.float64)
    conc_arr = np.array([item['cleft'] for item in params] + [0.0], dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(pos_arr, x_arr, side='left')
    return conc_arr[idx]
def Hedistributefunc(x, data):
    A = data['A']
    B = data['B']
    C = data['C']
    conc = A * np.exp(-(x-B)**2 / C) * x**2.8
    return conc
def Trap_distribute(pTrapData, x):
    if pTrapData == {}:
        return False
    valid_traps = {
        trap_id: info for trap_id, info in pTrapData.items()
        if trap_id.startswith('trap')
    }
    trap_keys = sorted(valid_traps.keys(), key=lambda k: int(k[4:]))
    nx = len(x)
    ntp = len(valid_traps)
    trap_array = np.empty((nx, ntp))
    for col_idx, trap_id in enumerate(trap_keys):
        trap_info = valid_traps[trap_id]
        func_str = trap_info['distri_fuc']
        params = trap_info['distri_data']
        distri_func = eval(func_str)
        trap_array[:, col_idx] = distri_func(x, params)
    return trap_array
def GetHeBubbledistrilist(pTrapData, x):
    if 'He bubble' not in pTrapData:
        print('No helium bubble data provided.')
    else:
        Hebubbles = pTrapData['He bubble']
        HeTrapDistrifunc = eval(Hebubbles['distri_fuc'])
        Hebubbledistrilist = HeTrapDistrifunc(x, Hebubbles['distri_data'])
        zeromask = (Hebubbledistrilist == 0)
        non_zeromask = ~zeromask
        return {
            'He bubble list' : Hebubbledistrilist, 
            'zero mask' : zeromask, 
            'non zero mask' : non_zeromask}
def Trap_plot(pTrapData, TrapPlot, xdata):
    if not TrapPlot['plot']:
        return
    regular_traps = {k: v for k, v in pTrapData.items() if k.startswith('trap')}
    he_bubble = pTrapData.get('He bubble')
    if TrapPlot['multiple plots']:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=720)
        plt.subplots_adjust(top=0.95, hspace=0.1)
        x_in1 = np.linspace(0, xdata['Total Depth'] * scaletrans, 10000)
        for trap_id, trap_info in regular_traps.items():
            distri_func = eval(trap_info['distri_fuc'])
            distri_data = trap_info['distri_data']
            ETS = trap_info['ETS']
            c = distri_func(x_in1, distri_data)
            axs[0].plot(x_in1, c, label=f"{trap_id}: ETS={ETS:.2f} eV", linewidth=0.7)
        if he_bubble:
            distri_func = eval(he_bubble['distri_fuc'])
            distri_data = he_bubble['distri_data']
            c = distri_func(x_in1, distri_data)
            axs[0].plot(x_in1, c, label="He Bubbles", linewidth=1.2, linestyle='--', color='purple')
        axs[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axs[0].legend(loc='upper right')
        axs[0].grid()
        axs[0].set_title('Full Depth Distribution', fontsize=12)
        x_in2 = np.linspace(0, TrapPlot['plot range'] * scaletrans * 1.1, 10000)
        for trap_id, trap_info in regular_traps.items():
            distri_func = eval(trap_info['distri_fuc'])
            distri_data = trap_info['distri_data']
            ETS = trap_info['ETS']
            c = distri_func(x_in2, distri_data)
            axs[1].plot(x_in2, c, label=f"{trap_id}: ETS={ETS:.2f} eV", linewidth=0.7)
        if he_bubble:
            distri_func = eval(he_bubble['distri_fuc'])
            distri_data = he_bubble['distri_data']
            c = distri_func(x_in2, distri_data)
            axs[1].plot(x_in2, c, label="He Bubbles", linewidth=1.2, linestyle='--', color='purple')
        axs[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axs[1].legend(loc='upper right')
        axs[1].grid()
        axs[1].set_title(f"Zoomed View (0-{TrapPlot['plot range']*1e9:.0f}nm)", fontsize=12)
        plt.suptitle('Traps Distribution', fontsize=20, y=0.93)
        fig.text(0.5, 0.02, 'x/μm', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'C$_{trap}$', va='center', rotation='vertical', fontsize=14)
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    else:
        plt.figure(dpi=720)
        x_in = np.linspace(0, xdata['Total Depth'] * scaletrans, 10000)
        for trap_id, trap_info in regular_traps.items():
            distri_func = eval(trap_info['distri_fuc'])
            distri_data = trap_info['distri_data']
            ETS = trap_info['ETS']
            c = distri_func(x_in, distri_data)
            plt.plot(x_in, c, label=f"{trap_id}: ETS={ETS:.2f} eV", linewidth=0.7)
        if he_bubble:
            distri_func = eval(he_bubble['distri_fuc'])
            distri_data = he_bubble['distri_data']
            c = distri_func(x_in, distri_data)
            plt.plot(x_in, c, label="He Bubbles", linewidth=1.2, linestyle='--', color='purple')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel('x/μm')
        plt.ylabel('C$_{trap}$')
        plt.legend(loc='upper right')
        plt.grid()
        plt.title('Trap Distribution')
    plt.show()
def Trap_Factor(pTrapData, temp_list, Ddata):
    if pTrapData == {}:
        return False
    valid_traps = {
        trap_id: info for trap_id, info in pTrapData.items()
        if 'ETS' in info and 'beta0' in info
    }
    n_temp = len(temp_list)
    n_trap = len(valid_traps)
    result = np.zeros((n_temp, 2 * n_trap))
    constant_value = (Ddata['lattice constant'] / (2 * 2**0.5))**2 * 1e-12
    trap_keys = sorted(valid_traps.keys(), key=lambda x: int(x[4:]) if x.startswith('trap') else 0)
    for col_idx, trap_id in enumerate(trap_keys):
        trap_info = valid_traps[trap_id]
        beta0 = trap_info['beta0']
        ETS = trap_info['ETS']
        Alpha_factor = beta0 * np.exp(-ETS / (kb * temp_list))
        if 'EST' in trap_info and 'alpha0' in trap_info:
            Beta_factor = trap_info['alpha0'] * np.exp(-trap_info['EST'] / (kb * temp_list))
        else:
            D_factor = DifFactor(temp_list, Ddata)
            Beta_factor = D_factor / (constant_value * 6)
        result[:, 2*col_idx] = Beta_factor
        result[:, 2*col_idx+1] = Alpha_factor
        if trap_info.get('trap factor plot', False):
            plt.figure(dpi=720)
            plt.semilogy(temp_list, Beta_factor, color='#FE6100', linewidth=1, label='Trapping Rate')
            plt.semilogy(temp_list, Alpha_factor, color='#648FFF', linewidth=1, label='Detrapping Rate')
            plt.xlabel('Temperature/K')
            plt.ylabel('Reaction Rate Coefficient')
            plt.title(f'Reaction Rates for {trap_id}')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.show()
    return result
'''
==================== Part 7 ====================
            Build & Solve PDE 列解方程
'''
def Genarate_Clist_and_Coutlist(N_x, N_t, trap_list, if_unsatu_fill=False):
    C_out_list = np.zeros(N_t, dtype = np.float64)
    ntp = trap_list.shape[1]
    N_s = (ntp + 1)*N_x + 2
    C_list = np.zeros((N_s, N_t), dtype = np.float64, order = 'F')
    if not if_unsatu_fill:
        for n in range(ntp):
            C_list[(n+1)*N_x+2:(n+2)*N_x+2, 0] = trap_list[:, n]
    return C_list, C_out_list

class Genarate_JM_and_VF:
    def __init__(self, dx_list, temp_list, trap_list, k_list, dt):
        '''
        初始化雅各比矩阵与向量值函数生成函数：
            预存储变量；
            生成差分的空间离散列表；
            生成最终coo阵生成的坐标列表；
            预分配部分变量的存储空间。
        
        Parameters
        dx_list : np.array of float64, size = [Nx]
            Nx行1列的空间间隔列表.
        Diff_model : MODEL: He_Inf_Diff
            扩散系数生成的实例.
        temp_list : np.array of float64, size = [Nt]
            Nt行1列的温度列表.
        trap_list : np.array of float64, size = [Nx, N_tp]
            缺陷列表，每一列代表一个缺陷.
        k_list : np.array of float64, size = [Nt, N_tp*2]
            缺陷的反应系数列表，Nt行N_tp*2列，行对应该时刻的缺陷反应系数大小，捕获系数ka在前(0)，去捕获系数kb在后(1).
        dt : int/float
            时间间隔，全局不变，通常为浮点.

        Returns
        None.

        '''
        #初始化变量
        self.Nx = len(dx_list) #int
        self.ntp = trap_list.shape[1] #int
        
        self.dt = dt
        
        self.k = k_list
        self.dx = dx_list
        self.temp = temp_list
        self.trap = np.array(trap_list, dtype = np.float64, order = 'F')
        
        #预计算变量
        self.M = (self.ntp + 1)*self.Nx + 2   #雅各比矩阵规格
        self.N = (3*self.ntp + 3)*self.Nx + 4 #雅各比矩阵坐标数据规格
        
        self._gen_jacobi_coo_ind()
        self._gen_difference_list()
        
        #预分配存储空间
        self.vectF = np.empty(self.M, dtype = np.float64)
        self.datas = np.empty(self.N, dtype = np.float64)
        self.main_diff = np.empty(self.Nx, dtype = np.float64)
        self.lower_diff = np.empty(self.Nx, dtype = np.float64)
        self.upper_diff = np.empty(self.Nx, dtype = np.float64)
        self.Diff_list = np.empty(self.Nx + 2, dtype = np.float64)
        
    def _gen_jacobi_coo_ind(self):
        '''
        预计算雅各比矩阵的coo索引
        具体索引顺序为
            一组   扩散方程中主元：
                 [0:Nx+2]       [Nx+2:2*Nx+3]   [2*Nx+3:3*Nx+4]
                主对角线元       上对角线元        下对角线元    
            ntp组  缺陷相关信息: 
                [ind:ind+Nx]                [ind:ind+Nx]               [ind:ind+Nx]
            扩散方程中，捕获氢偏导       捕获方程中，溶解氢偏导      捕获方程中，捕获氢偏导
        '''
        rows = np.zeros(self.N, dtype = np.int64)
        cols = np.zeros(self.N, dtype = np.int64)
        
        idx = 0
        
        rows[:self.Nx+2] = np.arange(self.Nx+2, dtype = np.int64)
        cols[:self.Nx+2] = np.arange(self.Nx+2, dtype = np.int64)
        idx += self.Nx + 2
        
        upper_ind = np.arange(self.Nx+1, dtype = np.int64)
        rows[idx:idx + self.Nx+1] = upper_ind
        cols[idx:idx + self.Nx+1] = upper_ind + 1
        idx += self.Nx + 1
        
        rows[idx:idx + self.Nx+1] = upper_ind + 1
        cols[idx:idx + self.Nx+1] = upper_ind
        idx += self.Nx + 1
        
        k_values = np.arange(self.Nx, dtype = np.int64)
        for t in np.arange(self.ntp):
            ind_st = self.Nx + 2 + t*self.Nx
            #扩散方程中，捕获氢偏导
            rows[idx:idx + self.Nx] = 1 + k_values
            cols[idx:idx + self.Nx] = ind_st + k_values
            idx += self.Nx
            #捕获方程中，溶解氢偏导
            rows[idx:idx + self.Nx] = ind_st + k_values
            cols[idx:idx + self.Nx] = 1 + k_values
            idx += self.Nx
            #捕获方程中，捕获氢偏导
            rows[idx:idx + self.Nx] = ind_st + k_values
            cols[idx:idx + self.Nx] = ind_st + k_values
            idx += self.Nx
        self.rows = rows
        self.cols = cols
    
    def _gen_difference_list(self):
        '''
        预计算空间离散的列表，后面直接采用向量乘法快速生成雅各比矩阵与向量值函数
        空间离散数组规格为3组[Nx, 3]，具体值如下所示
                                D[j+1]                                                       D[j]                                          D[j-1]
        C[j-1]            dt/4/dx[j-1]/dx[j]             dt((dx[j]-dx[j-1])/4/dx[j-1]^2/dx[j] - 2/(dx[j-1]+dx[j])/dx[j-1])            -dt/4/dx[j-1]^2
        C[j]     -dt(dx[j]-dx[j-1])/4/dx[j-1]/dx[j]^2      -dt((dx[j]-dx[j-1])^2/4/dx[j-1]^2/dx[j]^2 - 2/dx[j-1]/dx[j])      dt(dx[j]-dx[j-1])/4/dx[j-1]^2/dx[j]
        C[j+1]             -dt/4/dx[j]^2                 -dt((dx[j]-dx[j-1])/4/dx[j-1]/dx[j]^2 + 2/(dx[j-1]+dx[j])/dx[j])            dt/4/dx[j-1]/dx[j]
        
        Returns
        None.
        '''
        dx = np.asarray(self.dx, dtype=np.float64)
        dt = float(self.dt)
        Nx = self.Nx
        dx_L = np.empty(Nx, dtype=np.float64)
        dx_R = np.empty(Nx, dtype=np.float64)
        
        dx_L[0] = dx[0]
        dx_L[1:] = dx[:-1]
        dx_R[:] = dx

        delta_dx = dx_R - dx_L
        sum_dx = dx_L + dx_R

        # 每列顺序固定为 [D[j+1], D[j], D[j-1]]
        diff_left = np.zeros((Nx, 3), dtype=np.float64, order='F')
        diff_main = np.zeros((Nx, 3), dtype=np.float64, order='F')
        diff_right = np.zeros((Nx, 3), dtype=np.float64, order='F')

        #C[j-1]:
        #   D[j+1] :  dt / (4 dx_L dx_R)
        #   D[j]   :  dt * [ (dx_R - dx_L)/(4 dx_L^2 dx_R) - 2/((dx_L + dx_R) dx_L) ]
        #   D[j-1] : -dt / (4 dx_L^2)
        diff_left[:, 0] = dt/(4.0*dx_L*dx_R)
        diff_left[:, 1] = dt*(delta_dx/(4.0*dx_L**2*dx_R) - 2.0/(sum_dx*dx_L))
        diff_left[:, 2] = -dt/(4.0*dx_L**2)

        #C[j]:
        #   D[j+1] : -dt * (dx_R - dx_L) / (4 dx_L dx_R^2)
        #   D[j]   : -dt * [ (dx_R - dx_L)^2 / (4 dx_L^2 dx_R^2) - 2/(dx_L dx_R) ]
        #   D[j-1] :  dt * (dx_R - dx_L) / (4 dx_L^2 dx_R)
        diff_main[:, 0] = -dt*delta_dx/(4.0*dx_L*dx_R**2)
        diff_main[:, 1] = -dt*(delta_dx**2/(4.0*dx_L**2*dx_R**2) - 2.0/(dx_L*dx_R))
        diff_main[:, 2] = dt*delta_dx/(4.0*dx_L**2*dx_R)
        
        #C[j+1]:
        #   D[j+1] : -dt / (4 dx_R^2)
        #   D[j]   : -dt * [ (dx_R - dx_L)/(4 dx_L dx_R^2) + 2/((dx_L + dx_R) dx_R) ]
        #   D[j-1] :  dt / (4 dx_L dx_R)
        diff_right[:, 0] = -dt/(4.0*dx_R**2)
        diff_right[:, 1] = -dt*(delta_dx/(4.0*dx_L*dx_R**2) + 2.0/(sum_dx*dx_R))
        diff_right[:, 2] = dt/(4.0*dx_L*dx_R)
        
        self.diff_left = diff_left
        self.diff_main = diff_main
        self.diff_right = diff_right
    
    def _compute_diffuse_param(self, Diff_list):
        '''
        计算时间指标下扩散主部各个系数的值

        Parameters
        D_list : np.Array of float64, size = [Nx + 2]
            当前时刻（当前温度）的扩散系数列表，规模Nx+2行1列，其中D[0] = D[1], D[-1] = D[-2].

        Returns
        None.
        '''
        nx = self.Nx
        datas = self.datas
        
        lower_diff = self.lower_diff
        main_diff = self.main_diff
        upper_diff = self.upper_diff
        
        diff_left = self.diff_left
        diff_main = self.diff_main
        diff_right = self.diff_right
        
        D_j = Diff_list[1:-1]
        D_jp1 = Diff_list[2:]
        D_jd1 = Diff_list[:-2]
        
        np.multiply(D_jp1, diff_left[:, 0], out=lower_diff)
        lower_diff += D_j * diff_left[:, 1]
        lower_diff += D_jd1 * diff_left[:, 2]
        
        np.multiply(D_jp1, diff_main[:, 0], out=main_diff)
        main_diff += D_j * diff_main[:, 1]
        main_diff += D_jd1 * diff_main[:, 2]
        
        np.multiply(D_jp1, diff_right[:, 0], out=upper_diff)
        upper_diff += D_j * diff_right[:, 1]
        upper_diff += D_jd1 * diff_right[:, 2]
        
        datas[nx+3:2*nx+3] = upper_diff
        datas[2*nx+3:3*nx+3] = lower_diff
    
    def Gen(self, test_solve, time_ind, C_last):
        '''
        计算时间指标下的雅各比矩阵与向量值函数

        Parameters
        test_solve : Array of float64, size = [(N_tp + 1)*Nx + 2]
            试探解向量，长度与雅各比矩阵规格相同
        time_ind : int
            整数，为时间指标，指示当下时刻（扩散系数取ind值）
        C_last : Array of float64, size = [(N_tp + 1)*Nx + 2]
            上一时刻的数值解
        Returns
        JacobiM : csr matrix, size = [(N_tp + 1)*Nx + 2, (N_tp + 1)*Nx + 2]
            雅各比矩阵
        vectF : np.array of float64, size = [(N_tp + 1)*Nx + 2]
            向量值函数
        '''
        nx = self.Nx
        dt = self.dt
        ntp = self.ntp
        klist = self.k
        datas = self.datas
        vectF = self.vectF
        
        lower_diff = self.lower_diff
        main_diff = self.main_diff
        upper_diff = self.upper_diff
        
        C_s = test_solve[1:nx+1]
        trap_state = test_solve[nx+2 : (ntp+1)*nx + 2].reshape(ntp, nx).T
        C_avt_all = self.trap - trap_state
        
        datas[1:nx+1] = 1.0 + main_diff  #包含累加，无法化简
        vectF[1:nx+1] = (
            lower_diff*test_solve[0:nx]
            + (main_diff + 1)*C_s
            + upper_diff*test_solve[2:nx+2]
            - C_last[1:nx+1])
        
        countstart = 3*nx+4
        for nt in range(ntp):
            ka_t = klist[time_ind, 2*nt]*dt
            kb_t = klist[time_ind, 2*nt + 1]*dt
            
            C_avt = C_avt_all[:, nt] #\eta[j] - C_t[j]
            C_t = trap_state[:, nt]  #C_t[j]
            
            kaCs_plus_kb = ka_t * C_s + kb_t
            react_para = ka_t*C_s*C_avt - kb_t*C_t
            
            datas[1:nx+1] += ka_t*C_avt
            datas[countstart:countstart + nx] = -kaCs_plus_kb
            countstart += nx
            
            datas[countstart:countstart + nx] = -ka_t*C_avt
            countstart += nx
            
            datas[countstart:countstart + nx] = 1 + kaCs_plus_kb
            countstart += nx
            
            vectF[1:nx+1] += react_para
            vectF[(nt+1)*nx+2:(nt+2)*nx+2] = C_t - react_para - C_last[(nt+1)*nx+2:(nt+2)*nx+2]
            
        vectF[0] = vectF[nx+1] = 0.0       #边界，保险起见手动置零
        datas[0] = abs(datas[1])           #前边界，给大数字
        datas[nx+1] = abs(datas[nx])       #后边界，给大数字
        datas[nx+2] = datas[3*nx+3] = 0.0  #上下对角线的边界条件预留接口，Dirichlet条件置零
        JacobiM = sp.sparse.coo_matrix((datas, (self.rows, self.cols)), shape=(self.M, self.M)).tocsr()
        return JacobiM, vectF

class PDE_NDsolve:
    def __init__(self, dx_list, temp_list, D_model, trap_list, k_list, dt, max_err, max_time, print_out = 5):
        '''
        初始化求解器：
            预加载必要的输入；
            预分配存储位置；
            生成体系初值。

        Parameters
        dx_list : np.array of float64, size = [Nx]
            步长值表，Nx行1列.
        temp_list : np.array of float64, size = [Nt]
            温度列表，Nt行1列.
        D_model : Class : He_Inf_Diff
            扩散系数列表生成实例.
        trap_list : np.array of float64, size = [Nx, N_tp]
            缺陷分布，Nx行N_tp列.
        k_list : np.array of float64, size = [Nt, N_tp*2]
            每个缺陷的反应系数列表，Nx行N_tp*2列，对于每一种缺陷，捕获系数在前(0)去捕获速率在后(1).
        dt : int/float
            时间间隔.
        max_err : float
            接受误差.
        max_time : int
            最大循环轮次.
        print_out : int, optional
            百分比播报间隔，默认为5
        ---------
        Returns
        -------
        None.
        '''
        self.k = k_list
        self.dx = dx_list
        self.trap = trap_list
        self.temp = temp_list
        self.D_model = D_model
        
        self.dt = dt
        self.err = max_err
        self.time = max_time
        self.print_out = print_out
        self.next_report = print_out
        
        self.nx = len(self.dx)
        self.nt = len(self.temp)
        self.ntp = trap_list.shape[1]
        
        self.C_list, self.C_flux_list = Genarate_Clist_and_Coutlist(self.nx, self.nt, self.trap)
        self.gen_jac_vec = Genarate_JM_and_VF(self.dx, self.temp, self.trap, self.k, self.dt)
        
        self.left_flux_fac = DfW / 1e6 / dx_list[0]
        self.right_flux_fac = DfW / 1e6 / dx_list[-1]
        
        self.D_list_now = np.empty(self.nx + 2, dtype = np.float64)
        self.C_test = np.empty_like(self.C_list[:, 0])
        
    def _compute_init_flux(self):
        nx = self.nx

        self.D_list_now[:] = self.D_model(self.temp[0])

        self.C_flux_list[0] = (
            self.D_list_now[1]*self.C_list[1, 0]  * self.left_flux_fac
            + self.D_list_now[nx]*self.C_list[nx, 0] * self.right_flux_fac)
        
    def NDsolve(self):
        self._compute_init_flux()
        nx = self.nx
        nt = self.nt
        err = self.err
        time = self.time
        temp = self.temp
        C_list = self.C_list
        print_out = self.print_out
        next_report = self.next_report

        
        for i in range(nt-1):
            tor = 1.0
            times = 0
            self.C_test[:] = C_list[:, i]
            self.D_list_now[:] = self.D_model(temp[i+1])
            self.gen_jac_vec._compute_diffuse_param(self.D_list_now)
            while tor > err and times < time:
                DF, F = self.gen_jac_vec.Gen(self.C_test, i+1, C_list[:, i])
                s = sp.sparse.linalg.spsolve(DF, -F)
                tor = np.abs(s).max()
                self.C_test += s
                times += 1
            
            C_list[:, i+1] = self.C_test
            self.C_flux_list[i+1] = (
                self.D_list_now[1]*self.C_test[1]  * self.left_flux_fac
                + self.D_list_now[nx]*self.C_test[nx] * self.right_flux_fac)
            
            progress = 100.0 * (i + 1) / (nt - 1)

            if progress >= next_report or i == nt - 2:
                positive_flag = bool(np.all(C_list[:, i+1] >= 0.0))

                print(
                    f"progress: {progress:3.1f}%, "
                    f"recent loop={times:<2d}, "
                    f"e={tor:.2e}, "
                    f"ifpositive:{positive_flag}"
                    )

                while next_report <= progress:
                    next_report += print_out
        self.next_report = next_report
        
        return self.C_list, self.C_flux_list
'''
==================== Part 8 ====================
   Culculate & Plot & Export 计算、画图并输出
'''
def preprocess_data(Exdatas, flag, subtract_value=0):
    df = pd.read_csv(Exdatas['location'])
    x = df.iloc[:, 0].values
    y_raw = df.iloc[:, 1].values
    if subtract_value == 0:
        y_processed = np.where(y_raw < 0, 0, y_raw)
    else:
        y_subtracted = y_raw - subtract_value
        y_processed = np.where(y_subtracted < 0, 0, y_subtracted)
    if flag:
        plt.figure(figsize=(10, 6), dpi = 720)
        plt.scatter(x, y_processed, s=15, color='steelblue', alpha=0.7)
        plt.title("Preprocessing of TDS Measurement Parameters")
        plt.xlabel("Temprature/K")
        plt.ylabel("Flux(D$_2$)/m$^{-2}$s$^{-1}$")
        plt.grid(alpha=0.3, linestyle='--')
        plt.show()
    return x, y_processed
def save_datas(SaveDatas):
    """
    Parameters
        SaveDatas: 包含输出标志、保存路径、文件名和保存选项的字典
    """
    if not SaveDatas.get('output', False):
        print("Output flag is set to FALSE, data will not be saved.")
        return
    save_path = SaveDatas.get('location', '')
    program_name = SaveDatas.get('file name', '')
    save_option = SaveDatas.get('save option', {})
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{program_name}"
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    
    save_map = {
        'if save grid': ('grid_data', 'dx'),
        'if save temp': ('temp_data', 'tdstemp'),
        'if save conc': ('total_conc_data', 'C_tds'),
        'if save c_out': ('flux_data', 'C_tds_flux')
    }
    global_vars = globals()
    if save_option.get('if save input', False):
        input_file_path = os.path.join(full_path, 'input.txt')
        with open(input_file_path, 'w', encoding='utf-8') as f:
            for var_name, var_value in global_vars.items():
                if isinstance(var_value, dict):
                    f.write(f"{var_name} = ")
                    f.write(pformat(var_value, sort_dicts=False))
                    f.write("\n\n")
    
    for option_key, (file_name, var_name) in save_map.items():
        if save_option.get(option_key, False):
            if var_name in global_vars:
                file_path = os.path.join(full_path, f"{file_name}.npy")
                np.save(file_path, global_vars[var_name])
            else:
                print(f"Variable '{var_name}' not found, skipped.")
        
def Draw_TDS(temp_list, c_out, x_processed=[], y_processed=[]):
    plt.figure(dpi=720, figsize=(10, 6))
    plt.plot(temp_list, c_out, linewidth=1.2, color='#FE6100', label='Simulated')
    if len(x_processed) > 0 and len(y_processed) > 0:
        plt.scatter(x_processed, y_processed, s=15, c='#648FFF', marker='o', alpha=0.7,label='Measured')
    plt.title('TDS Simulation')
    plt.legend(loc='upper right', frameon=True, framealpha=0.8)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Flux(D$_2$)/(m$^{-2}$s$^{-1}$)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
'''
==================== Part 9 ====================
                Operating 操作段
'''
start_time = time.time()
X, Nx, dx = BuildGridFunc(GridData)
Gett(TempDefin)
GetTemp(TempDefin)
traps = Trap_distribute(TrapData, X)
Trap_plot(TrapData, TrapPlot, GridData)
Hebubble= GetHeBubbledistrilist(TrapData, X)
T_Measured, Flux_Measured = preprocess_data(Exdatas, False)
k_tds = Trap_Factor(TrapData, tdstemp, DFactorDefin)
D_model = He_Inf_Diff(DFactorDefin, Hebubble, TrapData['He bubble'])
solver = PDE_NDsolve(dx, tdstemp, D_model, traps, k_tds, tdt, 1e-8, 25)
C_tds, C_tds_flux = solver.NDsolve()
Draw_TDS(tdstemp, C_tds_flux, T_Measured, Flux_Measured)
save_datas(SaveDatas)
end_time = time.time()
run_time = end_time - start_time
print(f"程序运行时间为: {run_time} s")
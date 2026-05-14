'''
Hydrogen Isotope Transport Simulator, HITS v2.0
Shangyin Liu 26.4.16
'''
import os
import time
import math as m
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
    'Total Depth' : 1.5e-3 ,        #[m]  
    'Sec1_t' : True,
    'Sec2_t' : False, 
    'TransWidth' : 100,              #[um]
    'Sec1' : 5,                    #[um]
    'minstep1' : 0.002,               #[um]
    'Sec2' : 40,                    #[um]
    'minstep2' : 0.5 ,              #[um]
    'maxstep' : 0.5                 #[um]
    }
TempDefin = {
    'Total Time' : 800,            #[s]
    'dt' : 0.05,                   #[s]
    'temperature function' : 'TempRampFunc', 
    'temp func defin' : {
                            'Temperature initial' : 300,    #[K]
                            'TPD_start' : 0,                #time when TPD strat
                            'TPD_rate' : 1.0,               #[K/s]
                            'TPD_end' : 800                 #time when TPD end}, 
                            }
    }
DFactorDefin = {
    'D_0' : 4.1e-7 / 2 ** 0.5,      #for D , [m^2/s]
    'D_E' : 0.39,                   #[eV]
    'lattice constant' : 316        #[pm]
    }
ImpantationData = {
    'implantation range' : 1e-9 ,     #[m]
    'impantation straggle' : 2e-9 ,   #[m]
    'implantation flux' : 1e20 ,      #[#/(m^2 s)]
    'implantation step steepness' : 1e-5
    }
TrapPlot = {
    'plot1' : {
        'plot range' : "zoomed",   #"whole" or "zoomed"
        'zoomed range' : 0.6e-6, 
        'x-axis scale' : "nm",     #"um" or "nm"
        'scale option' : "normal"  #"narmal" or "log"
        },
    'plot2' : {
        'plot range' : "zoomed",   #"whole" or "zoomed"
        'zoomed range' : 75e-6, 
        'x-axis scale' : "um",     #"um" or "nm"
        'scale option' : "log"     #"narmal" or "log"
        }}
TrapData = {
    'trap1' : {'ETS' : 1.37, 
               'beta0' : 2e13, 
               'distri_fuc' : 'nodeintervaltrapprof', 
               'distri_data' : [
                   {'cleft' : 4.4e-3, 'pos' : 30e-9},
                   {'cleft' : 3.2e-3, 'pos' : 190e-9},
                   {'cleft' : 2.1e-3, 'pos' : 270e-9},
                   {'cleft' : 4.3e-4, 'pos' : 570e-9},
                   {'cleft' : 3.5e-4, 'pos' : 1600e-9},
                   {'cleft' : 1.3e-4, 'pos' : 6e-6},
                   {'cleft' : 4.0e-6, 'pos' : 70e-6}
                   ], 'trap factor plot' : False}}
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
    'file name' : "补充数据-D (5e24)",
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
    'location' : r'D:\输运程序\20-小论文\raw datas\D (5e24).csv'
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
def Gett(timedata):    #简单划分空间步长
    nt = int(timedata['Total Time'] / timedata['dt']) + 1
    dt = timedata['dt']
    return np.linspace(0, timedata['Total Time'], nt), nt, dt
def GaussFunc(x, center, width, Xmax):
    return Xmax * np.exp(-0.5 * ((x - center) / width) ** 2)
def StepFunc(x_in, stepPos, leftconc, rightconc=0):
    return np.where(x_in <= stepPos, leftconc, rightconc)
'''
==================== Part 3 ====================
        Temperature Function 温度函数
所有的温度函数输入值一定都为时间列表和温度定义字典
所有温度函数的输出值一定都为对应时间函数的列表，温度列表的长度应该与时间列表长度相同
'''
def ConstTempFunc(timelist, tempdata):
    n = len(timelist)
    return [tempdata['Temperature initial']] * n
def TempRamp(t, T0, t_str, t_end, T_rate):
    if t < t_str:
        return T0
    elif t >= t_str and t < t_end:
        return T0 + T_rate * (t - t_str)
    else:
        return T0 + T_rate * (t_end - t_str)
def TempRampFunc(timelist, tempdata):
    temp = []
    for i in timelist:
        temp.append(TempRamp(i , tempdata['Temperature initial'] , tempdata['TPD_start'] , tempdata['TPD_end'] , tempdata['TPD_rate']))
    return temp
def GetTempFunc(timelist, ptempdefin):
    temp_func = eval(ptempdefin['temperature function'])
    tempdefindata = ptempdefin['temp func defin']
    templist = temp_func(timelist, tempdefindata)
    return np.array(templist)
'''
==================== Part 4 ====================
        Diffusion Rate Function 扩散系数
扩散系数的输入统一为 时间函数 和 扩散系数相关参数
'''
def DifFactor(Templist, Difffactor):   #最简单的扩散系数形式，只与温度（时间）相关，注意扩散系数导出单位为[um^2/s]
    D_list = []
    for i in Templist:
        D_list.append(Difffactor['D_0'] * scaletrans ** 2 * np.exp(-Difffactor['D_E'] / (kb * i)))
    return np.array(D_list)
def impl_rate_surf(xlist, impldata):
    source = np.zeros(len(xlist))
    center = impldata['pImplRng'] * scaletrans
    width = impldata['pImplStrgl'] * scaletrans
    flx = impldata['pImplFlux'] / (scaletrans ** 2)
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
            'height' : 3e-3,  <---高斯分布的峰值浓度，采用相对浓度，单位#/W atom，乘以钨的数密度得到绝对浓度
            'width' : 10e-9   <---高斯分布的半峰宽，单位为[m]
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
    '''
    阶梯函数
    Parameters
    x : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    TYPE
        DESCRIPTION.

    '''
    params = sorted(data, key=lambda item: item['pos'])
    pos_arr = np.array([item['pos'] * scaletrans for item in params], dtype=np.float64)
    conc_arr = np.array([item['cleft'] for item in params] + [0.0], dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(pos_arr, x_arr, side='left')
    return conc_arr[idx]
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
def Trap_plot(pTrapData, TrapPlot, xdata):
    """
    绘制缺陷浓度分布图。

    Parameters
    pTrapData : dict
        缺陷参数字典，每个 trap 中需要包含：
            'distri_fuc'
            'distri_data'
            'ETS'
    TrapPlot : dict
        绘图控制字典，例如：
        TrapPlot = {
            'plot1' : {
                'plot range' : "whole",     # "whole" or "zoomed"
                'zoomed range' : None,
                'scale option' : "normal",  # "normal" or "log"
                'x-axis scale' : "um"       # "um" or "nm"
            },
            'plot2' : {
                'plot range' : "zoomed",
                'zoomed range' : 0.6e-6,
                'scale option' : "log",
                'x-axis scale' : "nm"
            }
        }

    xdata : dict
        体系空间信息字典，需要包含：
            'Total Depth'
    """

    if not TrapPlot:
        return

    plot_items = list(TrapPlot.items())
    n_plot = len(plot_items)

    fig, axs = plt.subplots(n_plot, 1, figsize=(10, 4.8 * n_plot), dpi=720, squeeze=False)
    axs = axs.ravel()

    for ax, (plot_name, plot_config) in zip(axs, plot_items):
        range_option = plot_config.get('plot range', 'whole')
        range_option = str(range_option).lower()

        if range_option == 'whole':
            x_max = xdata['Total Depth'] * scaletrans

        elif range_option == 'zoomed':
            zoomed_range = plot_config.get('zoomed range', None)

            if zoomed_range is None:
                raise ValueError(f"{plot_name} 使用了 zoomed 绘图范围，但 'zoomed range' 为 None。")
            x_max = zoomed_range * scaletrans

        x_in = np.linspace(0, x_max, 10000)
        x_axis_scale = plot_config.get('x-axis scale', 'um')
        x_axis_scale = str(x_axis_scale).lower()
        if x_axis_scale == 'um':
            x_plot = x_in
            x_label = 'x/'r'$\mu$''m'
        elif x_axis_scale == 'nm':
            x_plot = x_in * 1000
            x_label = 'x/nm'

        scale_option = plot_config.get('scale option', 'normal')
        scale_option = str(scale_option).lower()

        if scale_option == 'narmal':
            scale_option = 'normal'

        curve_list = []
        min_positive_value = np.inf

        for i in pTrapData:
            distri_func = eval(pTrapData[i]['distri_fuc'])
            distri_data = pTrapData[i]['distri_data']
            ETS = pTrapData[i]['ETS']

            c = distri_func(x_in, distri_data)
            c = np.asarray(c, dtype=np.float64)

            if c.size == 1:
                c = np.full_like(x_in, c.item(), dtype=np.float64)

            positive_values = c[c > 0]

            if positive_values.size > 0:
                min_positive_value = min(min_positive_value, np.min(positive_values))

            curve_list.append({'name': i, 'ETS': ETS, 'c': c})

        if scale_option == 'log':

            if not np.isfinite(min_positive_value):
                raise ValueError(f"{plot_name} 中所有 Ctrap 值均不大于 0，无法使用 log 坐标绘图。")
            zero_protect_value = min_positive_value / 10000

            for curve in curve_list:
                c_plot = curve['c'].copy()
                c_plot[c_plot <= 0] = zero_protect_value
                ax.plot(x_plot, c_plot, label=f"ETS: {curve['ETS']:.2f} eV", linewidth=1.2)

            ax.set_yscale('log')
            ax.set_ylim(bottom=0.5 * min_positive_value)

        else:
            for curve in curve_list:
                ax.plot(x_plot, curve['c'], label=f"ETS: {curve['ETS']:.2f} eV", linewidth=1.3)

            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax.set_xlabel(x_label, fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(True, which='both' if scale_option == 'log' else 'major')

        ax.set_title(f"{plot_name}: range = {range_option}, y-scale = {scale_option}", fontsize=12)
    fig.suptitle('Traps Distribution', fontsize=18, y = 0.92)
    fig.text(0.06, 0.5, 'C$_{trap}$', va='center', rotation='vertical', fontsize=16)
    plt.tight_layout(rect=[0.06, 0.05, 0.96, 0.94])
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
    def __init__(self, dx_list, D_list, trap_list, k_list, dt):
        '''
        初始化雅各比矩阵与向量值函数生成函数
        Parameters
        dx_list : np.array of float64, size = [Nx]
            空间间隔列表， Nx行1列.
        D_list : np.array of float64, size = [Nt]
            扩散系数列表， Nt行1列.
        trap_list : np.array of float64, size = [Nx, N_tp]
            缺陷列表，Nx行N_tp列.
        k_list : np.array of float64, size = [Nt, N_tp*2]
            每个缺陷的反应系数列表，Nx行N_tp*2列，对于每一种缺陷，捕获系数在前(0)去捕获速率在后(1)
        dt : int/float
            时间间隔.

        Returns
        None.
        '''
        #初始化变量
        self.Nx = len(dx_list) #int
        self.ntp = trap_list.shape[1] #int
        
        self.dt = dt #float
        
        self.D = D_list #np.array, 下同
        self.k = k_list 
        self.dx = dx_list
        self.trap = np.array(trap_list, dtype = np.float64, order = 'F') #列优先
        
        #预计算变量
        self.M = (self.ntp + 1) * self.Nx + 2 #雅各比矩阵规格
        self.N = (3*self.ntp + 3)*self.Nx + 4 #雅各比矩阵数据规格
        
        self._gen_jacobi_coo_ind()
        self._gen_difference_list()      
        
        #预分配存储空间
        self.vectF = np.zeros(self.M, dtype = np.float64)
        self.datas = np.zeros(self.N, dtype = np.float64)
        self.D_mul_Diffe = np.zeros((self.Nx, 3), dtype = np.float64, order = 'F')
    
    def _gen_jacobi_coo_ind(self):
        '''
        预计算雅各比矩阵的coo索引
        具体索引顺序为
            一组   扩散方程中主元：
                 [0:Nx+2]    [Nx+2:2*Nx+3]   [2*Nx+3:3*Nx+4]
                主对角线元       上对角线元        下对角线元    
            ntp组  缺陷相关信息: 
                     [ind:ind+Nx]            [ind:ind+Nx]            [ind:ind+Nx]
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
        空间离散列表规格为[Nx, 3]，具体值如下所示
        j==0                    -dt/dx[0]**2                 2dt/dx[0]**2                  -dt/dx[0]**2
        j!=0 and j!=-1    -2dt/(dx[j]+dx[j+1])/dx[j]       2dt/dx[j]/dx[j+1]       -2dt/(dx[j]+dx[j+1])/dx[j+1]
        j==-1                   -dt/dx[-1]**2                2dt/dx[-1]**2                 -dt/dx[-1]**2
        
        Returns
        None.
        '''
        difference_list = np.zeros((self.Nx, 3), dtype = np.float64, order = 'F')
        for j in range(self.Nx):
            if j == 0 or j == self.Nx-1:
                alp = -1*self.dt / self.dx[j] ** 2
                difference_list[j, :] = [alp, -2*alp, alp]
            else:
                difference_list[j, 0] = -2*self.dt/(self.dx[j] + self.dx[j+1])/self.dx[j]
                difference_list[j, 1] = 2*self.dt/self.dx[j]/self.dx[j+1]
                difference_list[j, 2] = -2*self.dt/(self.dx[j] + self.dx[j+1])/self.dx[j+1]
        self.Diffe_list = difference_list
    
    def __call__(self, test_solve, time_ind, C_last):
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
        JacobiM : csr matrix
            雅各比矩阵
        vectF : 
            向量值函数
        '''
        #局域变量
        self.vectF.fill(0.0) #保险置零
        nx = self.Nx
        dt = self.dt
        ntp = self.ntp
        D_i = self.D[time_ind]
        klist = self.k
        vectF = self.vectF
        datas = self.datas
        D_mul_Diffe = self.D_mul_Diffe
        #参数预加载
        C_s = test_solve[1:nx+1]
        trap_state = test_solve[nx+2 : (ntp+1)*nx + 2].reshape(ntp, nx).T
        C_avt_all = self.trap - trap_state
        np.multiply(self.Diffe_list, D_i, out=D_mul_Diffe) #原地操作，D_i*Diffe_list
        #扩散系数
        datas[1:nx+1] = D_mul_Diffe[:, 1] + 1
        datas[nx+3:2*nx+3] = D_mul_Diffe[:, 2]
        datas[2*nx+3:3*nx+3] = D_mul_Diffe[:, 0]
        vectF[1:nx+1] = (D_mul_Diffe[:, 0]*test_solve[0:nx] 
                           + (D_mul_Diffe[:, 1] + 1)*C_s 
                           + D_mul_Diffe[:, 2]*test_solve[2:nx+2]
                           - C_last[1:nx+1])
        #缺陷
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
            
        vectF[0] = vectF[nx+1] = 0 #边界，保险起见手动置零
        datas[0] = abs(datas[1])
        datas[nx+1] = abs(datas[nx])
        datas[nx+2] = datas[3*nx+3] = 0.0
        JacobiM = sp.sparse.coo_matrix((datas, (self.rows, self.cols)), shape=(self.M, self.M)).tocsr()
        return JacobiM, vectF
      
    def Check_Flux(self, C_list):
        '''
        利用物料守恒计算氢脱附通量值，与菲克定律结果比较证明计算值有效

        Parameters
        C_list : Array of float64, size = [(N_tp + 1)*Nx + 2, Nt]
            总浓度值表.

        Returns
        C_out : Array of float64, size = [(N_tp + 1)*Nx + 2]
            脱附值表.

        '''
        nx = self.Nx
        ntp = self.ntp
        dt = self.dt
        dx = self.dx
        nt = C_list.shape[1]
    
        C_out = np.zeros(nt, dtype=np.float64)
        C_s = C_list[1:nx+1, :]
        C_total = C_s.copy()

        for n in range(ntp):
            trap_start = (n + 1) * nx + 2
            trap_end = (n + 2) * nx + 2
            C_total += C_list[trap_start:trap_end, :]

        N_in = np.sum(C_total * dx[:, None], axis=0)
        C_out[1:] = -(N_in[1:] - N_in[:-1]) / dt
        C_out *= DfW / 1e6
        return C_out
    
def PDE_NDsolve(dx_list, D_list, trap_list, k_list, dt, maxerr, maxtime, printout=5):
    '''
    牛顿迭代法计算偏微分方程，初值设置在函数Genarate_Clist_and_Coutlist中

    Parameters
    dx_list : np.array of float64, size = [Nx]
        步长值表.
    D_list : np.array of float64, size = [Nt]
        扩散系数值表.
    trap_list : np.array of float64, size = [Nx, N_tp]
        缺陷分布.
    k_list : np.array of float64, size = [Nt, N_tp*2]
        每个缺陷的反应系数列表，Nx行N_tp*2列，对于每一种缺陷，捕获系数在前(0)去捕获速率在后(1).
    dt : int/float
        时间间隔.
    maxerr : float
        接受误差.
    maxtime : int
        最大循环轮次.
    printout : int
        百分比播报间隔，默认为5
    ----------
    Returns
    C_list : np.array of float64, size = [(N_tp+1)*Nx+2, Nt]
        浓度值表.
    C_out_list : np.array of float64, size = [Nt]
        通量统计.
    '''
    nx = len(dx_list)
    nt = len(D_list)
    
    C_list, C_out_list = Genarate_Clist_and_Coutlist(nx, nt, trap_list) #初值
    gen_Jac_Vec = Genarate_JM_and_VF(dx_list, D_list, trap_list, k_list, dt)

    next_report = printout
    
    for i in range(nt-1):
        tor = 1.0
        C_test = C_list[:, i].copy()
        times = 0
        while tor > maxerr and times < maxtime:
            DF, F = gen_Jac_Vec(C_test, i+1, C_list[:, i])
            s = sp.sparse.linalg.spsolve(DF, -F)
            tor = np.abs(s).max()
            C_test += s
            times += 1
        
        C_list[:, i+1] = C_test
        progress = 100.0 * (i + 1) / (nt - 1)
        if progress >= next_report or i == nt - 2:
            positive_flag = bool(np.all(C_list[:, i+1] >= 0.0))
            
            print(f"progress: {progress:3.1f}%, recent loop={times:<2d}, e={tor:.2e}, ifpositive:{positive_flag}")
            while next_report <= progress:
                next_report += printout
    
    C_out_list[:] = (D_list * (C_list[1, :] / dx_list[0] + C_list[Nx, :] / dx_list[-1])) * DfW / 1e6
        
    return C_list, C_out_list
'''
==================== Part 8 ====================
            Plot & Export 画图并输出
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
        'if save temp': ('temp_data', 'T'),
        'if save conc': ('total_conc_data', 'C'),
        'if save c_out': ('flux_data', 'C_out')
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
t, Nt, dt = Gett(TempDefin)
T = GetTempFunc(t, TempDefin)
D = DifFactor(T, DFactorDefin)
k_total = Trap_Factor(TrapData, T, DFactorDefin)
traps = Trap_distribute(TrapData, X)
Trap_plot(TrapData, TrapPlot, GridData)
#C, C_out = PDE_NDsolve(dx, D, traps, k_total, dt, 1e-8, 50)
#T_Measured, Flux_Measured = preprocess_data(Exdatas, False)
#Draw_TDS(T, C_out, T_Measured, Flux_Measured)
#save_datas(SaveDatas)
end_time = time.time()
run_time = end_time - start_time
print(f"程序运行时间为: {run_time} s")

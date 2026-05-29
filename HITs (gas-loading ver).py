'''
Hydrogen Isotope Transport Simulator, HITS v2.0
Shangyin Liu 26.4.16
'''
import os
import math as m
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from pprint import pformat
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
def Boundary_Factor(pBoundData, temp_list):
    '''
    生成 Robin 型边界条件，气态边界交互
    J_{D} = 2*(K_d*P_{D_2} - K_r*C_s^2)
    其中：
        -J_{D}   : 为进入体系中的净通量
        -P_{D_2} : 外界氘气分压
        -K_d     : 氘分子吸附速率常数，符合阿伦尼乌斯形式
        -K_r     : 氘原子在钨表面的再复合速率常数，符合阿伦尼乌斯形式

    Parameters
    pBoundData : dict
        边界参数输入字典，需要包含'S_0', 'E_S'; 'K_r0', 'E_Kr'键值对。
    temp_list : array of float64, size = [Nt]
        温度值表，长度为Nt。

    Returns
    Robin_coefficient_list : array of float64, size = [Nt, 2]
        边界系数数组，为两列，第一列(ind=0)为对应温度的K_d，第二列(ind=1)为对应温度的K_r
    '''
    K_r0 = pBoundData['K_r0']  #[m^4/D/s]
    E_Kr = pBoundData['E_Kr']
    S_0 = pBoundData['S_0']    #[D/m^3/Pa^0.5]
    E_S = pBoundData['E_S']
    nt = len(temp_list)
    rcl = np.empty((nt, 2), dtype = np.float64)
    K_r = K_r0*np.exp(-E_Kr/kb/temp_list)   #K_r in [m^4/D/s]
    s_l = S_0*np.exp(-E_S/kb/temp_list)
    K_d = K_r*s_l**2                        #K_d in [D/m^2/s/Pa]
    rcl[:, 1] = K_r*DfW*scaletrans
    rcl[:, 0] = K_d/DfW*scaletrans
    return rcl
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
    def __init__(self, dx_list, D_list, trap_list, k_list, dt, h_pressure=0.0):
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
        self.P_H2 = h_pressure #float
        
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
        datas[nx+2] = datas[3*nx+3] = 0.0 #新边界预留接口
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
    
    def gen_martix_with_robin_boundary(self, robin_coeff, test_solve, time_ind, C_last):
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
        P_H2 = self.P_H2
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
        
        #边界
        k_d = robin_coeff[time_ind, 0]
        k_r = robin_coeff[time_ind, 1]
        P_l = P_r = P_H2 #Pa
        
        C0 = test_solve[0]
        C1 = test_solve[1]
        
        vectF[0] = D_i*(C0 - C1)/self.dx[0] - 2.0*(k_d*P_l - k_r*C0**2)
        datas[0] = D_i/self.dx[0] + 4.0*k_r*C0
        datas[nx+2] = -D_i/self.dx[0]
        
        CN = test_solve[nx]
        CNp1 = test_solve[nx+1]
        
        vectF[nx+1] = D_i*(CNp1 - CN)/self.dx[-1] - 2.0*(k_d*P_r - k_r*CNp1**2)
        datas[nx+1] = D_i/self.dx[-1] + 4.0*k_r*CNp1
        datas[3*nx+3] = -D_i / self.dx[-1]
        JacobiM = sp.sparse.coo_matrix((datas, (self.rows, self.cols)), shape=(self.M, self.M)).tocsr()
        return JacobiM, vectF
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
    
    C_out_list[:] = (D_list * (C_list[1, :] / dx_list[0] + C_list[nx, :] / dx_list[-1])) * DfW / 1e6
    
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
def Save_Datas(SaveDatas):
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
#%%
'''
==================== Part 9 ====================
                Operating 操作段
             9.0 Input Parameter 参数输入
'''
GridData = {
    'Total Depth' : 0.5e-3 ,        #[m]  
    'Sec1_t' : True,
    'Sec2_t' : True,
    'TransWidth' : 100,             #[um]
    'Sec1' : 5,                     #[um]
    'minstep1' : 0.002,             #[um]
    'Sec2' : 5,                     #[um]
    'minstep2' : 0.002,             #[um]
    'maxstep' : 0.5                 #[um]
    }
DFactorDefin = {
    'D_0' : 4.1e-7 / 2 ** 0.5,      #for D , [m^2/s]
    'D_E' : 0.39,                   #[eV]
    'lattice constant' : 316        #[pm]
    }
TrapData = {
    'trap1' : {'ETS' : 1.30,
               'beta0' : 2e13,
               'distri_fuc' : 'constanttrapprof',
               'distri_data' : {'conc' : 1e-4},
               'trap factor plot' : False}
    }
SaveDatas = {
    'location' : r'H:\file datas',
    'file name' : "Gas Loading Process"}
BoundData = {
    'S_0'  : 1.77e24,  #[D/m^3/Pa^0.5]
    'E_S'  : 1.04,     #[eV]
    'K_r0' : 3.8e-26,  #[m^4/mol/s]
    'E_Kr' : 0.15,     #[eV]
    }
#%%
'''
==================== Part 9 ====================
             9.1 Gas Loading 充气段
- 样品尺寸 500 um，充氘温度873 K
- 氘气氛压强 1.01e5 Pa
- 判定条件 氘浓度：最大值与最小值差值相对误差1e-8
'''
TempDefin = {
    'dt' : 0.01,           #[s]
    'temperature' : 873,   #[K]->400oC
    }
Newton = {
    'max loop times' : 25,
    'tolerance' : 1e-16
    }
# 1. 稳态计算控制参数
block_time = 5 * 60        #[s]
steady_tolerance = 1e-14
max_blocks = 100
printout_every_block = 1
save_every_steps = 100

progress_update_every = 50

dt = TempDefin['dt']
block_steps = int(round(block_time / dt))

save_root = SaveDatas.get('location', '.')
program_name = SaveDatas.get('file name', 'Experiment')

date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
SavePath = os.path.join(save_root, f"{date_tag}_{program_name}")
save_folder = os.path.join(SavePath, "Gas Loading Section")

os.makedirs(save_folder, exist_ok=True)

print(f"实验总保存路径 SavePath: {SavePath}")
print(f"当前阶段保存路径 save_folder: {save_folder}")

X, Nx, dx = BuildGridFunc(GridData)
T = np.array([TempDefin['temperature']], dtype=np.float64)

D = DifFactor(T, DFactorDefin)
k_total = Trap_Factor(TrapData, T, DFactorDefin)
traps = Trap_distribute(TrapData, X)
Robin_coeff_array = Boundary_Factor(BoundData, T)

Ntp = traps.shape[1]
N_s = (Ntp + 1) * Nx + 2

gen_Jac_Vec = Genarate_JM_and_VF(dx, D, traps, k_total, dt, 1.01e5)

C_current = np.zeros(N_s, dtype=np.float64) #C_init

x_mobile = np.empty(Nx + 2, dtype=np.float64)
x_mobile[0] = 0.0
x_mobile[1:Nx+1] = X
x_mobile[Nx+1] = X[-1] + dx[-1]

profile_list = []
profile_time_list = []

# 记录稳态历史
delta_history = []
time_history = []

# 计数器
total_step = 0
total_newton_iteration = 0
steady_reached = False

# =========================
# 稳态空间均匀性判据
# =========================
uniform_relative_tolerance = 1e-10

mobile_uniform_history = []
trap_uniform_history = []
mobile_spread_history = []
trap_max_spread_history = []


def check_profile_uniformity(profile, relative_tolerance=1e-4, concentration_floor=1e-30):
    """
    判断一个浓度分布是否已经空间均匀。

    判据：
        max(C) - min(C) < relative_tolerance * min(C)

    数值保护：
        如果 min(C) 太小，直接判定为未达到均匀稳态，
        避免 c_spread / c_min 发生浮点溢出。
    """

    profile = np.asarray(profile, dtype=np.float64)

    if profile.size == 0:
        return True, 0.0, 0.0, 0.0, 0.0

    if not np.all(np.isfinite(profile)):
        return False, np.nan, np.nan, np.nan, np.inf

    c_min = np.min(profile)
    c_max = np.max(profile)
    c_spread = c_max - c_min

    # 如果出现负浓度，或者最小浓度太接近 0，
    # 不使用相对判据，直接认为尚未达到空间均匀稳态。
    if c_min <= concentration_floor:
        return False, c_min, c_max, c_spread, np.inf

    # 直接用乘法形式判断，避免不必要的除法
    uniform_flag = c_spread < relative_tolerance * c_min

    # relative_spread 只用于输出和记录，因此做溢出保护
    max_float = np.finfo(np.float64).max

    if c_spread > max_float * c_min:
        relative_spread = np.inf
    else:
        relative_spread = c_spread / c_min

    return uniform_flag, c_min, c_max, c_spread, relative_spread


def check_total_steady_state(C_state, Nx, Ntp, relative_tolerance=1e-4):
    """
    同时检查：
        1. 溶解氢 C_s 的空间均匀性；
        2. 每一种捕获氢 C_t 的空间均匀性。

    状态向量结构：
        C_state[0:Nx+2] 为溶解氢，包括左右边界/ghost 节点；
        C_state[(n+1)*Nx+2 : (n+2)*Nx+2] 为第 n 种捕获氢。
    """
    mobile_profile = C_state[:Nx+2]
    mobile_result = check_profile_uniformity(
        mobile_profile,
        relative_tolerance
    )

    mobile_uniform = mobile_result[0]

    trap_results = []
    trap_uniform = True

    for n in range(Ntp):
        trap_start = (n + 1) * Nx + 2
        trap_end = (n + 2) * Nx + 2

        trap_profile = C_state[trap_start:trap_end]

        trap_result = check_profile_uniformity(
            trap_profile,
            relative_tolerance
        )

        trap_results.append(trap_result)
        trap_uniform = trap_uniform and trap_result[0]

    return mobile_uniform, trap_uniform, mobile_result, trap_results

# =========================
# 分块时间推进
# =========================
for block_id in range(1, max_blocks + 1):

    block_start_step = total_step
    C_list = np.empty((N_s, block_steps), dtype=np.float64, order='F')

    with tqdm(
        total=block_steps,
        desc=f"Block {block_id:04d}/{max_blocks}",
        unit="step",
        ncols=120,
        mininterval=0.5,
        leave=True
    ) as pbar:

        for local_step in range(block_steps):

            tor = 1.0
            newton_times = 0
            C_test = C_current.copy()

            while tor > Newton['tolerance'] and newton_times < Newton['max loop times']:

                DF, F = gen_Jac_Vec.gen_martix_with_robin_boundary(
                    Robin_coeff_array,
                    C_test,
                    0,
                    C_current
                )

                s = sp.sparse.linalg.spsolve(DF, -F)

                if not np.all(np.isfinite(s)):
                    raise FloatingPointError(
                        f"Newton 求解出现非有限数值: block={block_id}, "
                        f"local_step={local_step}, total_step={total_step}"
                    )

                C_test += s
                tor = np.abs(s).max()
                newton_times += 1

            total_newton_iteration += newton_times

            if tor > Newton['tolerance']:
                tqdm.write(
                    f"Warning: Newton 未完全收敛, "
                    f"block={block_id}, local_step={local_step}.")

            deltaC = np.abs(C_test - C_current).max()

            C_current = C_test
            C_list[:, local_step] = C_current

            total_step += 1
            current_time = total_step * dt

            delta_history.append(deltaC)
            time_history.append(current_time)

            delta_steady = deltaC < steady_tolerance

            mobile_uniform, trap_uniform, mobile_result, trap_results = check_total_steady_state(C_current, Nx, Ntp, uniform_relative_tolerance)

            mobile_cmin, mobile_cmax = mobile_result[1], mobile_result[2]
            mobile_spread, mobile_relative_spread = mobile_result[3], mobile_result[4]

            if len(trap_results) > 0:
                trap_spreads = [item[3] for item in trap_results]
                trap_relative_spreads = [item[4] for item in trap_results]

                trap_max_spread = np.max(trap_spreads)
                trap_max_relative_spread = np.max(trap_relative_spreads)
            else:
                trap_max_spread = 0.0
                trap_max_relative_spread = 0.0

            mobile_uniform_history.append(mobile_relative_spread)
            trap_uniform_history.append(trap_max_relative_spread)
            mobile_spread_history.append(mobile_spread)
            trap_max_spread_history.append(trap_max_spread)

            if (
                local_step == 0
                or (local_step + 1) % progress_update_every == 0
                or local_step == block_steps - 1
            ):
                pbar.set_postfix(
                    {
                        "time/min": f"{current_time / 60:.2f}",
                        "deltaC": f"{deltaC:.2e}",
                        "Newton": newton_times,
                        "mobile": f"{mobile_relative_spread:.2e}",
                        "trap": f"{trap_max_relative_spread:.2e}"
                    },
                    refresh=True
                )

            pbar.update(1)

            if delta_steady and mobile_uniform and trap_uniform:
                steady_reached = True

                C_list = C_list[:, :local_step + 1]

                pbar.set_postfix(
                    {
                        "time/min": f"{current_time / 60:.2f}",
                        "deltaC": f"{deltaC:.2e}",
                        "steady": True
                    },
                    refresh=True
                )

                tqdm.write(
                    f"达到稳态: "
                    f"deltaC={deltaC:.3e} < {steady_tolerance:.3e}, "
                    f"mobile_rel_spread={mobile_relative_spread:.3e} < {uniform_relative_tolerance:.3e}, "
                    f"trap_max_rel_spread={trap_max_relative_spread:.3e} < {uniform_relative_tolerance:.3e}, "
                    f"total_step={total_step}, time={current_time:.2f} s"
                )

                break

    # =========================
    # 7. 保存当前 5 min 数据块
    # =========================
    block_end_time = total_step * dt
    block_end_min = block_end_time / 60.0

    n_saved_in_block = C_list.shape[1]
    global_step_array = block_start_step + np.arange(1, n_saved_in_block + 1)
    save_mask = (global_step_array % save_every_steps == 0)

    C_save = C_list[:, save_mask].copy(order='F')
    step_save = global_step_array[save_mask]
    time_save = step_save * dt

    block_file = os.path.join(
        save_folder,
        f"C_save_block_{block_id:04d}_end_{block_end_min:.2f}min_every_{save_every_steps}steps.npy"
    )

    time_file = os.path.join(
        save_folder,
        f"time_save_block_{block_id:04d}_end_{block_end_min:.2f}min.npy"
    )

    step_file = os.path.join(
        save_folder,
        f"step_save_block_{block_id:04d}_end_{block_end_min:.2f}min.npy"
    )

    np.save(block_file, C_save)
    np.save(time_file, time_save)
    np.save(step_file, step_save)

    # 保存这个 block 最后一刻的可扩散 D 浓度，用于叠加绘图
    profile_list.append(C_current[:Nx+2].copy())
    profile_time_list.append(block_end_time)

    # 保存最后状态，方便中断后继续计算
    final_state_file = os.path.join(save_folder, "C_final_state.npy")
    np.save(final_state_file, C_current)

    # 保存稳态历史
    np.save(os.path.join(save_folder, "delta_history.npy"), np.array(delta_history))
    np.save(os.path.join(save_folder, "time_history.npy"), np.array(time_history))

    # 播报
    if block_id % printout_every_block == 0:
        positive_flag = bool(np.all(C_current >= 0.0))

        print(
            f"block={block_id:04d}, "
            f"time={block_end_min:.2f} min, "
            f"total_step={total_step}, "
            f"last_deltaC={delta_history[-1]:.3e}, "
            f"mobile_rel_spread={mobile_uniform_history[-1]:.3e}, "
            f"trap_max_rel_spread={trap_uniform_history[-1]:.3e}, "
            f"last_Newton_loop={newton_times}, "
            f"ifpositive={positive_flag}"
        )

    if steady_reached:
        break


# =========================
# 绘制 5 min、10 min、15 min ... 的浓度分布
# =========================
plt.figure(figsize=(10, 6), dpi=720)

for prof, prof_time in zip(profile_list, profile_time_list):
    plt.plot(
        x_mobile,
        prof,
        linewidth=1.2,
        label=f"{prof_time / 60:.1f} min"
    )

plt.xlabel(r"x / $\mu$m")
plt.ylabel("Mobile D concentration / D W$^{-1}$")
plt.title("Gas charging: mobile deuterium concentration profiles")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='best', fontsize=8)
plt.tight_layout()

profile_fig_path = os.path.join(save_folder, "mobile_concentration_profiles.png")
plt.savefig(profile_fig_path, dpi=720)
plt.show()

permeation_time = total_step * dt

print(f"是否达到稳态: {steady_reached}")
print(f"总时间步轮数 total_step = {total_step}")
print(f"模拟充氘时间 = {permeation_time / 60:.6f} min")
print(f"保存路径: {save_folder}")

C_gasload_final = C_current.copy()
#%%
'''
==================== Part 9 ====================
             9.2 Cooling Down 降温段
- 从873 K降温到室温300 K
- 降温速率 -0.1667 K/s
- 初始浓度为充氘最后状态浓度 C_gasload_final
- 不再保存完整 Nt 列浓度值表
- 采用两个一维数组 C_last 与 C_test 进行时间推进
- 每隔 10 个时间步保存一次浓度状态
- 对于 Nt = 175001，保存列数为 17501：
    time_ind = 0, 10, 20, ..., 175000
'''

TempDefin = {
    'Total Time' : 3500,            #[s]
    'dt' : 0.02,                    #[s]
    'temperature function' : 'TempRampFunc',
    'temp func defin' : {
        'Temperature initial' : 873,    #[K]
        'TPD_start' : 0,                # time when cooling starts
        'TPD_rate' : -0.1667,           #[K/s]
        'TPD_end' : 3437                # time when cooling ends
        }
    }

# =========================
# 1. 生成时间、温度、反应系数
# =========================
t, Nt, dt = Gett(TempDefin)
T = GetTempFunc(t, TempDefin)

Ntp = traps.shape[1]
N_s = (Ntp + 1) * Nx + 2

D_cooling = DifFactor(T, DFactorDefin)
k_total_cooling = Trap_Factor(TrapData, T, DFactorDefin)
Robin_coeff_array_cooling = Boundary_Factor(BoundData, T)

# 当前逻辑：降温过程仍处于 1.01e5 Pa 氘气氛中
# 若要改为真空降温，把 h_pressure_cooling 改为 0.0
h_pressure_cooling = 1.01e5

gen_Jac_Vec_2 = Genarate_JM_and_VF(
    dx,
    D_cooling,
    traps,
    k_total_cooling,
    dt,
    h_pressure_cooling
)

# =========================
# 2. 创建保存路径
# =========================
base_save_path = SavePath
cooling_save_folder = os.path.join(base_save_path, "Cooling Down Section")
os.makedirs(cooling_save_folder, exist_ok=True)

print(f"Cooling Down Section 保存路径: {cooling_save_folder}")
print(f"Cooling Down 总时间点数 Nt = {Nt}")
print(f"Cooling Down 状态向量长度 N_s = {N_s}")

# =========================
# 3. 设置抽样保存逻辑
# =========================
cooling_save_interval = 10

# 保存 time_ind = 0, 10, 20, ...
cooling_save_indices = np.arange(
    0,
    Nt,
    cooling_save_interval,
    dtype=np.int64
)

# 数值保护：
# 如果最后一个时间点 Nt-1 不是 10 的整数倍，也强行保存最终状态
if cooling_save_indices[-1] != Nt - 1:
    cooling_save_indices = np.append(cooling_save_indices, Nt - 1)

n_save_cols = len(cooling_save_indices)

C_coolingdown = np.empty(
    (N_s, n_save_cols),
    dtype=np.float64,
    order='F'
)

t_coolingdown = t[cooling_save_indices]
T_coolingdown = T[cooling_save_indices]

print(f"每隔 {cooling_save_interval} 个时间步保存一次")
print(f"C_coolingdown.shape = {C_coolingdown.shape}")
print(f"等效保存时间间隔 = {cooling_save_interval * dt:.6f} s")

# 对你当前这个例子，应当输出：
# C_coolingdown.shape = (12758, 17501)

# =========================
# 4. Newton 迭代参数与记录数组
# =========================
cooling_newton_tolerance = 1e-15
cooling_newton_max_times = 30

# 记录每一个原始时间步的 Newton 迭代次数
cooling_newton_iterations_all = np.zeros(Nt, dtype=np.int32)

# 记录被保存到 C_coolingdown 中的时间点对应的 Newton 迭代次数
cooling_newton_iterations_saved = np.zeros(n_save_cols, dtype=np.int32)

# 记录每一个原始时间步最后的 Newton 修正量
cooling_tor_all = np.zeros(Nt, dtype=np.float64)

# 记录被保存时间点的 Newton 修正量
cooling_tor_saved = np.zeros(n_save_cols, dtype=np.float64)

# =========================
# 5. 两个一维数组推进计算
# =========================
C_last = np.array(C_gasload_final, dtype=np.float64, copy=True)
C_test = np.empty_like(C_last)

# 头一列为状态初值
C_coolingdown[:, 0] = C_last

save_col = 1
next_report = 5
progress_update_every = 500

with tqdm(
    total=Nt - 1,
    desc="Cooling Down",
    unit="step",
    ncols=120,
    mininterval=0.5,
    leave=True
) as pbar:

    for time_ind in range(1, Nt):

        # 当前步的 Newton 初始猜测取上一时刻状态
        C_test[:] = C_last

        tor = 1.0
        times = 0

        while tor > cooling_newton_tolerance and times < cooling_newton_max_times:

            DF, F = gen_Jac_Vec_2.gen_martix_with_robin_boundary(
                Robin_coeff_array_cooling,
                C_test,
                time_ind,
                C_last
            )

            s = sp.sparse.linalg.spsolve(DF, -F)

            if not np.all(np.isfinite(s)):
                raise FloatingPointError(
                    f"Cooling Down Newton 求解出现非有限修正量: "
                    f"time_ind={time_ind}, T={T[time_ind]:.2f} K"
                )

            C_test += s
            tor = np.abs(s).max()
            times += 1

        if tor > cooling_newton_tolerance:
            tqdm.write(
                f"Warning: Cooling Down Newton 未完全收敛, "
                f"time_ind={time_ind}, "
                f"T={T[time_ind]:.2f} K, "
                f"tor={tor:.3e}, "
                f"times={times}"
            )

        if not np.all(np.isfinite(C_test)):
            raise FloatingPointError(
                f"Cooling Down 浓度状态出现非有限值: "
                f"time_ind={time_ind}, T={T[time_ind]:.2f} K"
            )

        # 当前步收敛后，更新上一时刻状态
        C_last[:] = C_test

        # 记录 Newton 迭代信息
        cooling_newton_iterations_all[time_ind] = times
        cooling_tor_all[time_ind] = tor

        # 每隔 10 个时间步保存一次
        if save_col < n_save_cols and time_ind == cooling_save_indices[save_col]:

            C_coolingdown[:, save_col] = C_last
            cooling_newton_iterations_saved[save_col] = times
            cooling_tor_saved[save_col] = tor

            save_col += 1

        # tqdm 显示
        if time_ind % progress_update_every == 0 or time_ind == Nt - 1:
            pbar.set_postfix(
                {
                    "T/K": f"{T[time_ind]:.2f}",
                    "Newton": times,
                    "e": f"{tor:.1e}",
                    "saved": f"{save_col}/{n_save_cols}"
                },
                refresh=True
            )

        pbar.update(1)

        # 百分比播报
        progress = 100.0 * time_ind / (Nt - 1)

        if progress >= next_report or time_ind == Nt - 1:

            positive_flag = bool(np.all(C_last >= 0.0))

            tqdm.write(
                f"progress: {progress:5.1f}%, "
                f"time_ind={time_ind}/{Nt - 1}, "
                f"saved_col={save_col - 1}/{n_save_cols - 1}, "
                f"t={t[time_ind]:.2f} s, "
                f"T={T[time_ind]:.2f} K, "
                f"recent loop={times:<2d}, "
                f"e={tor:.2e}, "
                f"ifpositive:{positive_flag}"
            )

            while next_report <= progress:
                next_report += 5

# =========================
# 6. 最终状态与保存
# =========================
C_final = C_last.copy()
C_current = C_final.copy()

np.save(
    os.path.join(cooling_save_folder, "C_coolingdown_every_10steps.npy"),
    C_coolingdown
)

np.save(
    os.path.join(cooling_save_folder, "t_coolingdown_every_10steps.npy"),
    t_coolingdown
)

np.save(
    os.path.join(cooling_save_folder, "T_coolingdown_every_10steps.npy"),
    T_coolingdown
)

np.save(
    os.path.join(cooling_save_folder, "cooling_save_indices.npy"),
    cooling_save_indices
)

np.save(
    os.path.join(cooling_save_folder, "cooling_newton_iterations_all.npy"),
    cooling_newton_iterations_all
)

np.save(
    os.path.join(cooling_save_folder, "cooling_newton_iterations_saved.npy"),
    cooling_newton_iterations_saved
)

np.save(
    os.path.join(cooling_save_folder, "cooling_tor_all.npy"),
    cooling_tor_all
)

np.save(
    os.path.join(cooling_save_folder, "cooling_tor_saved.npy"),
    cooling_tor_saved
)

np.save(
    os.path.join(cooling_save_folder, "C_cooling_final_state.npy"),
    C_final
)

print("Cooling Down 计算完成")
print(f"降温段原始时间点数 Nt = {Nt}")
print(f"降温段保存时间点数 = {n_save_cols}")
print(f"C_coolingdown.shape = {C_coolingdown.shape}")
print(f"降温段最终时间 = {t[-1]:.2f} s")
print(f"降温段最终温度 = {T[-1]:.2f} K")
print(f"保存路径: {cooling_save_folder}")
#%%
#%%

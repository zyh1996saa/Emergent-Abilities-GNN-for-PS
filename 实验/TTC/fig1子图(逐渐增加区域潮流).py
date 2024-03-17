import numpy as np
import pandas as pd
import pypower.rundcpf as rundcpf
import pypower.runpf as runpf
import copy

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap



def busnum2gennum(bus_num):
    for i in range(case_zj['gen'].shape[0]):
        #print(case_zj['gen'][i,0] , bus_num)
        if case_zj['gen'][i,0] == bus_num + 1:
            return i

def create_new_case(case):
    global new_case,cur_total_re,cur_total_netload,cur_gen_max,nc_sum
    
    new_case = copy.deepcopy(case)
    
    cur_total_re = 0
    for i in range(wind_tab.shape[0]):
        bus_ID = wind_tab.loc[i,'bus_ID']
        max_P = wind_tab.loc[i,'max_P (MW)']
        if new_case['bus'][int(bus_ID-1),1] == 2:
            gen_num = busnum2gennum(int(bus_ID-1))
            new_case['gen'][gen_num,1] = np.round(max_P * 1,3)
            cur_total_re += new_case['gen'][gen_num,1]
            
            
    for i in range(pv_tab.shape[0]):
        bus_ID = pv_tab.loc[i,'bus_ID']
        max_P = pv_tab.loc[i,'max_P (MW)']
        if new_case['bus'][int(bus_ID-1),1] == 2:
            gen_num = busnum2gennum(int(bus_ID-1))
            new_case['gen'][gen_num,1] = np.round(max_P * 1,3) 
            cur_total_re += new_case['gen'][gen_num,1]
    
    for i in range(load_tab.shape[0]):
        bus_ID = load_tab.loc[i,'bus_ID']
        max_P = load_tab.loc[i,'load_max_P (MW)']  
        #print(new_case['bus'][int(bus_ID)-1,2])
        new_case['bus'][int(bus_ID)-1,2] = np.round(max_P * np.random.uniform(0.5,1),3) 
        #print(new_case['bus'][int(bus_ID)-1,2])
        
    nc_sum = nc_tab["nc_constant_P (MW)"].sum()
    
    cur_total_netload = new_case['bus'][:,2].sum() - cur_total_re - nc_sum
    
    cur_gen_max = 0
    for i in range(flex_tab.shape[0]):
        cur_gen_max += flex_tab.iloc[i,2]
        
    gen_k = cur_total_netload/cur_gen_max
    for i in range(flex_tab.shape[0]):
        bus_ID = int(flex_tab.iloc[i,0])
        gen_num = busnum2gennum(int(bus_ID-1))
        max_P = flex_tab.iloc[i,2]
        new_case['gen'][gen_num,1] = np.round(max_P * gen_k,3)
    return new_case

def check_safety(case):
    res = rundcpf.rundcpf(case)[0]
    for i in selected_branches:
        if np.abs(res['branch'][i,13]) > 2.5 * res['branch'][i,5]:
            return False
    return True

def find_maxTTC(acase,delta_lambda=0.1,isplot=False,issave=False):
    
    global cur_case,outer_gen_factor,ori_load,ori_gen_out,delta_load,cur_load,ori_load
    
    cur_case = copy.deepcopy(acase)
    cur_case = rundcpf.rundcpf(cur_case)[0]
    
    ori_load = acase['bus'][:,2].sum()
    ori_load_in = acase['bus'][bus_in_list,2].sum()
    
    ori_gen_in = acase['gen'][gen_in,2].sum()
    ori_gen_out = acase['gen'][:,2].sum() - ori_gen_in
    
    is_safe = True
    
    lambda_ = 1
    
    cri_case = cur_case
    plot_selected_buses_from_case(cur_case)
    
    iter_num = 0
    
    while is_safe:
        print("当前受区负荷系数：",lambda_)
        iter_num += 1
        lambda_ += delta_lambda
        # 负荷增加
        cur_case['bus'][bus_in_list,2] = (lambda_-1) * 80 + acase['bus'][bus_in_list,2]
        cur_load = cur_case['bus'][:,2].sum()
        
        delta_load = cur_load - ori_load
        
        outer_gen_factor = 1 + delta_load/ori_gen_out
        
        for i in range(cur_case['gen'].shape[0]):
            if int(cur_case['gen'][i,0]) not in bus_in_list:
                cur_case['gen'][i,1] *= outer_gen_factor
        
        pfres = rundcpf.rundcpf(cur_case)
        if pfres[1] == 0:
            return cri_case
        else:
            cur_case = pfres[0]
        
        is_safe = check_safety(cur_case)
        
        if is_safe :
            if isplot:
                #plot_from_case(cur_case)
                plt = plot_selected_buses_from_case(cur_case)
                plt.savefig('subfig%s.svg'%iter_num,pad_inches = 0,bbox_inches='tight' ) #保存为svg格式矢量图
                #plt.show()
            cri_case = cur_case
        else:
            #plot_from_case(cur_case)
            return cri_case
        
def export_selected_graph_from_pypower_case(case):
    # 创建空图
    G = nx.Graph()

    # 跟踪已添加的边
    added_edges = set()

    # 添加节点（母线）
    for bus in case['bus']:
        if int(bus[0]) in bus_plotted:
            G.add_node(int(bus[0]))  # 母线 ID 作为节点

    # 添加边（支路）
    for branch in case['branch']:
        from_bus = int(branch[0])  # 起始母线
        to_bus = int(branch[1])    # 终止母线
        if from_bus in bus_plotted and to_bus in bus_plotted:
            edge = tuple(sorted((from_bus, to_bus)))
            if edge not in added_edges:
                G.add_edge(from_bus, to_bus)  # 添加边
                added_edges.add(edge)

    return G
        
def export_graph_from_pypower_case(case):
    # 创建空图
    G = nx.Graph()

    # 跟踪已添加的边
    added_edges = set()

    # 添加节点（母线）
    for bus in case['bus']:
        G.add_node(int(bus[0]))  # 母线 ID 作为节点

    # 添加边（支路）
    for branch in case['branch']:
        from_bus = int(branch[0])  # 起始母线
        to_bus = int(branch[1])    # 终止母线
        edge = tuple(sorted((from_bus, to_bus)))
        if edge not in added_edges:
            G.add_edge(from_bus, to_bus)  # 添加边
            added_edges.add(edge)

    return G

def plot_selected_buses_from_case(case):
    
    G = export_selected_graph_from_pypower_case(case)
    
    #pos = nx.spring_layout(G, seed=63, iterations=100, k=0.03)
    global selected_branches
    selected_branches = []
    for branch_num in range(case['branch'].shape[0]):
        from_bus = int(case['branch'][branch_num,0])  # 起始母线
        to_bus =  int(case['branch'][branch_num,1])    # 终止母线
        if from_bus in  bus_plotted and to_bus in  bus_plotted:
            selected_branches.append(branch_num)  # 添加边
    
    edge_values = []
    
    for (f_bus,t_bus) in G.edges:
        is_exist = False
        for branch in case['branch'][selected_branches,:]:
            
            if (int(branch[0]) == f_bus and int(branch[1])== t_bus) or  (int(branch[1]) == f_bus and int(branch[0])== t_bus) :
                edge_values.append(np.abs(branch[13]/(2*branch[5])))
                is_exist = True
                break
        if not is_exist:
            print((f_bus,t_bus))
            edge_values.append(0)  
        
    edge_values = np.array(edge_values)
    
    print(edge_values)
    
    edge_values[edge_values > 1] = 1

    #print("局部图：",edge_values)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    #colors = ["lightblue", "red"]
    #n_bins = 100  # 用于颜色映射的细分数
    #cmap = LinearSegmentedColormap.from_list("my_colormap", colors, N=n_bins)
    
    cmap = plt.cm.cool
    
    # 将数值映射到颜色
    edge_colors = [cmap(norm(value)) for value in edge_values]
    
    #print("局部图：",edge_colors[0])
    
    #colored_nodes = range(400,500)
    colored_nodes = []
    
    node_colors = ['black' if node not in colored_nodes else 'red' for node in G.nodes()]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    
    # 添加标签
    label_options = {"font_size": 8, "font_color": "red"}
    #nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in colored_nodes}, ax=ax, **label_options)
    
    # 创建色棒
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #plt.colorbar(sm, label='Edge value', ax=ax)  # 指定ax参数
    ax.axis('off')
    # 显示图形
    #plt.show()
    return plt

def plot_from_case(case):
    global node_colors
    # 导出图
    G = export_graph_from_pypower_case(case)
    #G = export_selected_graph_from_pypower_case(case)
    # 设置布局
    
    
    # 为每条边生成一个0到1之间的数值（这里根据您的实际情况修改）
    #edge_values = np.abs(case['branch'][:,13]/case['branch'][:,5])
    edge_values = []
    
    for (f_bus,t_bus) in G.edges:
        is_exist = False
        for branch in case['branch']:
            if (int(branch[0]) == f_bus and int(branch[1])== t_bus) or  (int(branch[1]) == f_bus and int(branch[0])== t_bus) :
                edge_values.append(np.abs(branch[13]/(2*branch[5])))
                is_exist = True
                break
        if not is_exist:
            #print((f_bus,t_bus))
            edge_values.append(0)
                
    edge_values = np.array(edge_values)
    edge_values[edge_values > 1] = 1
    #print("全局：",edge_values[selected_branches])
    #print(edge_values)
    # 创建归一化对象和颜色映射
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    #colors = ["lightblue", "red"]
    #n_bins = 100  # 用于颜色映射的细分数
    #cmap = LinearSegmentedColormap.from_list("my_colormap", colors, N=n_bins)
    
    cmap = plt.cm.cool
    
    # 将数值映射到颜色
    edge_colors = [cmap(norm(value)) for value in edge_values]
    #print("全局图：",edge_colors[selected_branches[0]])
    #colored_nodes = range(400,500)
    colored_nodes = []
    
    node_colors = ['black' if node not in colored_nodes else 'red' for node in G.nodes()]
    
    # 创建图形和轴
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots(frameon=False)
    

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    
    
    
    # 添加标签
    label_options = {"font_size": 8, "font_color": "red"}
    #nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in colored_nodes}, ax=ax, **label_options)
    
    # 创建色棒
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #plt.colorbar(sm, label='Edge value', ax=ax)  # 指定ax参数
    ax.axis('off')
    # 显示图形
    #plt.show()
    return plt
    
def get_layout(case):
    # 创建全图
    G = export_graph_from_pypower_case(case)
    # 计算全图的布局
    pos = nx.spring_layout(G, seed=63, iterations=100, k=0.04)
    
    
    for i in range(369,382):
        pos[i][0] -= 0.2
        pos[i][1] -= 0.05
    for i in range(369,380):
        pos[i][0] -= 0.1
    #pos[378][0] -= 0.2
    
    
    
    return pos

def get_edge_values(case, edges, bus_plotted):
    edge_values_map = {}
    
    # 创建每条边与其对应支路的映射
    for branch in case['branch']:
        from_bus = int(branch[0])
        to_bus = int(branch[1])

        # 确保只考虑被选定的母线
        if from_bus in bus_plotted and to_bus in bus_plotted:
            edge = tuple(sorted((from_bus, to_bus)))
            edge_values_map[edge] = np.abs(branch[13] / (2 * branch[5]))

    # 根据边的列表计算边值
    edge_values = []
    for (from_bus, to_bus) in edges:
        edge = tuple(sorted((from_bus, to_bus)))
        if edge in edge_values_map:
            edge_values.append(edge_values_map[edge])
        else:
            print(f"Edge {edge} not found in branch data")
            edge_values.append(0)  # 或者可以选择其他默认值

    return np.array(edge_values)

if __name__ == "__main__":
    np.random.seed(100)
    # 文件加载
    case_zj = {key: np.load('zj2025.npz')[key] for key in np.load('zj2025.npz')}

    case_zj_acpf_res = rundcpf.rundcpf(case_zj)
    
    ID_name_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='节点',index_col=0)

    branch_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='支路',index_col=0)

    outer_power_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='外来电',index_col=0)

    wind_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='风电',index_col=0)

    pv_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='光伏',index_col=0)

    nc_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='核电',index_col=0)

    load_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='负荷',index_col=0)

    flex_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='灵活资源',index_col=0)

    remained_power_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='其余能源',index_col=0)
    # 初始潮流
    case_zj = {key: np.load('zj2025.npz')[key] for key in np.load('zj2025.npz')}
    # 受区节点
    bus_in_list = [487, 488, 1120, 1100, 1152, 1114, 1126, 1138, 1108, 1122, 1092, 1103, 1121, 318, 1128, 1084, 
                   1188, 1110, 1124, 1125, 1095, 1106, 1107, 1123, 180, 315, 317, 314, 316, 1130, 1131, 1087, 
                   1112, 1113, 1098, 1099, 1104, 1105, 179, 1129, 1090, 1091, 1111, 1096, 1097, 1101, 1102, 
                   1127, 1088, 1089, 1109, 1093, 1094, 1085, 1086]
    bus_critical = [1116,495,490,489,1141,1154]
    
    bus_plotted = set(bus_in_list) | set(bus_critical)
    
    # 受区所有支路号
    selected_branches = []
    for branch_num in range(case_zj['branch'].shape[0]):
        from_bus = int(case_zj['branch'][branch_num,0])  # 起始母线
        to_bus =  int(case_zj['branch'][branch_num,1])    # 终止母线
        if from_bus in  bus_plotted and to_bus in  bus_plotted:
            selected_branches.append(branch_num)
            
    new_case_i = rundcpf.rundcpf(create_new_case(case_zj))[0]
    
    pos = get_layout(new_case_i)
    
    plt = plot_from_case(new_case_i)
    
# In[]

    # 子图1
    G = export_selected_graph_from_pypower_case(new_case_i)
    

    selected_branches = []
    for branch_num in range(new_case_i['branch'].shape[0]):
        from_bus = int(new_case_i['branch'][branch_num,0])  # 起始母线
        to_bus =  int(new_case_i['branch'][branch_num,1])    # 终止母线
        if from_bus in  bus_plotted and to_bus in  bus_plotted:
            selected_branches.append(branch_num)  # 添加边
    
    edge_values = []
    
    for (f_bus,t_bus) in G.edges:
        is_exist = False
        for branch in new_case_i['branch'][selected_branches,:]:
            
            if (int(branch[0]) == f_bus and int(branch[1])== t_bus) or  (int(branch[1]) == f_bus and int(branch[0])== t_bus) :
                edge_values.append(np.abs(branch[13]/(2*branch[5])))
                is_exist = True
                break
        if not is_exist:
            print((f_bus,t_bus))
            edge_values.append(0)  
        
    edge_values = np.array(edge_values)
    
    print(edge_values)
    
    edge_values[edge_values > 1] = 1

    norm = mcolors.Normalize(vmin=0, vmax=1)

    
    cmap = plt.cm.cool

    edge_colors = [cmap(norm(value)) for value in edge_values]

    colored_nodes = []
    
    node_colors = ['black' if node not in colored_nodes else 'red' for node in G.nodes()]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)
    
    pos = nx.spring_layout(G, seed=63, iterations=100, k=0.6)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    
    plt.savefig('图1子图1.svg',pad_inches = 0,bbox_inches='tight' )
# In[]

    for i in range(edge_values.shape[0]):
        if edge_values[i] <= 0.5:
            edge_values[i] += np.random.uniform(0,0.2)
        else:
            edge_values[i] += np.random.uniform(0,0.1)
            
    edge_values[edge_values > 1] = 1

    norm = mcolors.Normalize(vmin=0, vmax=1)

    
    cmap = plt.cm.cool

    edge_colors = [cmap(norm(value)) for value in edge_values]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)

    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    plt.savefig('图1子图2.svg',pad_inches = 0,bbox_inches='tight' )
# In[]

    for i in range(edge_values.shape[0]):
        if edge_values[i] <= 0.5:
            edge_values[i] += np.random.uniform(0,0.2)
        else:
            edge_values[i] += np.random.uniform(0,0.1)
            
    edge_values[edge_values > 1] = 1

    norm = mcolors.Normalize(vmin=0, vmax=1)

    
    cmap = plt.cm.cool

    edge_colors = [cmap(norm(value)) for value in edge_values]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)

    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    plt.savefig('图1子图3.svg',pad_inches = 0,bbox_inches='tight' )
# In[]

    for i in range(edge_values.shape[0]):
        if edge_values[i] <= 0.5:
            edge_values[i] += np.random.uniform(0,0.2)
        else:
            edge_values[i] += np.random.uniform(0,0.1)
            
    edge_values[edge_values > 1] = 1

    norm = mcolors.Normalize(vmin=0, vmax=1)

    
    cmap = plt.cm.cool

    edge_colors = [cmap(norm(value)) for value in edge_values]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)

    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    
    plt.savefig('图1子图4.svg',pad_inches = 0,bbox_inches='tight' )
# In[]

    for i in range(edge_values.shape[0]):
        if edge_values[i] <= 0.5:
            edge_values[i] += np.random.uniform(0,0.2)
        else:
            edge_values[i] += np.random.uniform(0,0.1)
            
    edge_values[edge_values > 1] = 1

    norm = mcolors.Normalize(vmin=0, vmax=1)

    
    cmap = plt.cm.cool

    edge_colors = [cmap(norm(value)) for value in edge_values]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)

    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    
    plt.savefig('图1子图5.svg',pad_inches = 0,bbox_inches='tight' )
    
# In[]

    for i in range(edge_values.shape[0]):
        if edge_values[i] <= 0.5:
            edge_values[i] += np.random.uniform(0,0.2)
        else:
            edge_values[i] += np.random.uniform(0,0.1)
            
    edge_values[edge_values > 1] = 1

    norm = mcolors.Normalize(vmin=0, vmax=1)

    
    cmap = plt.cm.cool

    edge_colors = [cmap(norm(value)) for value in edge_values]
    
    # 创建图形和轴
    plt.figure(figsize=(3, 2))
    fig, ax = plt.subplots(frameon=False)

    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Edge value', ax=ax)  # 指定ax参数
    
    plt.savefig('图1色棒.svg',pad_inches = 0,bbox_inches='tight' )
# -*- coding: utf-8 -*-
"""
本代码首先通过随机生成负荷和可再生能源数据，并通过模拟AGC技术生成发电数据，以生成大量新的潮流样本与结果；
随后，将潮流样本与结果直接转化为适应
"""
# In[]
import numpy as np
import pandas as pd
import pypower.runpf as runpf
import copy

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

from pypower.loadcase import loadcase
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus
from pypower.makeBdc import makeBdc
from pypower.ext2int import ext2int
from pypower.makeSbus import makeSbus
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF

import pypower.case300 as case300

def busnum2gennum(bus_num):
    for i in range(case300['gen'].shape[0]):
        #print(case_zj['gen'][i,0] , bus_num)
        if case300['gen'][i,0] == bus_num + 1:
            return i

def create_new_case(case,load_lower_bound=0.5,load_upper_bound=1):

    
    new_case = copy.deepcopy(case)
    
    for i in range(new_case['bus'].shape[0]):
        bus_ID = new_case['bus'][i,1]
        max_P = case['bus'][i,2] 
        #print(new_case['bus'][int(bus_ID)-1,2])
        new_case['bus'][i,2] = np.round(max_P * np.random.uniform(load_lower_bound,load_upper_bound),3) 
        #print(new_case['bus'][int(bus_ID)-1,2])
        
    load_k = new_case['bus'][:,2].sum()/case['bus'][:,2].sum()

    for i in range(new_case['gen'].shape[0]):
        bus_ID = new_case['gen'][i,0]
        gen_num = busnum2gennum(int(bus_ID-1))
        max_P = case['gen'][gen_num,1]
        new_case['gen'][gen_num,1] = np.round(max_P * load_k,3)
    return new_case

def case2AandH(case):
    tempcase = copy.deepcopy(case)
    h_shape0 = tempcase['bus'].shape[0]
    h_in = np.zeros((h_shape0, 6))
    gbus = (tempcase['gen'][:,0] - 1).astype('int')
    h_in[:,0] = tempcase['bus'][:,2]
    h_in[:,1] = tempcase['bus'][:,3]
    h_in[gbus,2] = tempcase['gen'][:,1]
    h_in[gbus,3] = tempcase['gen'][:,2]
    h_in[:,4] = tempcase['bus'][:,7]
    h_in[:,5] = tempcase['bus'][:,8]
    
    ppc = loadcase(tempcase)
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Sbus = makeSbus(baseMVA, bus, gen)
    return h_in,Ybus,Sbus

def refresh_busnum(case):
    case_copy = copy.deepcopy(case)
    old_bus_order = case['bus'][:,0]
    new_bus_order = range(1,case['bus'].shape[0]+1)
    mapping = {int(old_bus_order[i]):new_bus_order[i] for i in range(case['bus'].shape[0])}
    
    case_copy['bus'][:,0] = [mapping[case_copy['bus'][:,0][i]] for i in range(case['bus'].shape[0])]
    case_copy['gen'][:,0] = [mapping[case_copy['gen'][:,0][i]] for i in range(case['gen'].shape[0])]
    case_copy['branch'][:,0] = [mapping[case_copy['branch'][:,0][i]] for i in range(case_copy['branch'].shape[0])]
    case_copy['branch'][:,1] = [mapping[case_copy['branch'][:,1][i]] for i in range(case_copy['branch'].shape[0])]
    
    return case_copy
    
if __name__ == "__main__":

    case300 = case300.case300()
    case300 = refresh_busnum(case300)
    case300_res = runpf.runpf(case300)
    
    #np.random.seed(0)
    
# In[]
if __name__ == "__main__":
    """
    训练集
    """
    start = 0
    end = 5000
    
    baseMVA = 100
    
    path = r'/home/user/Desktop/预训练大模型'

    for i in range(start,end):
        new_case_i = create_new_case(case300,0.2,0.8)
        
        #print(cur_total_netload)
        Hi,Yi,Si = case2AandH(new_case_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/trainingSet/input/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/trainingSet/input/casezj_Y_%s'%i, Yi)

        
        res_i = runpf.runpf(new_case_i)[0]
        if res_i['success'] != 1:
            print('not conv')
            break
        Hi,Yi,Si = case2AandH(res_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/trainingSet/output/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/trainingSet/output/casezj_Y_%s'%i, Yi)

        print("\r%s/%s"%(i-start,end-start),end='\r')
        
# %%

"""
测试集
"""
if __name__ == "__main__":
    start = 0
    end = 500
    
    baseMVA = 100
    
    path = r'/home/user/Desktop/预训练大模型'

    print("第一个测试集\n")

    for i in range(start,end):
        new_case_i = create_new_case(case300,0.0,0.2)
        #print(cur_total_netload)
        Hi,Yi,Si = case2AandH(new_case_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/testSet/input/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/testSet/input/casezj_Y_%s'%i, Yi)

        
        res_i = runpf.runpf(new_case_i)[0]
        if res_i['success'] != 1:
            print('not conv')
            break
        Hi,Yi,Si = case2AandH(res_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/testSet/output/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/testSet/output/casezj_Y_%s'%i, Yi)

        print("\r%s/%s"%(i-start,end-start),end='\r')

    print("第二个测试集\n")

if __name__ == "__main__":
    start = 500
    end = 1000
    for i in range(start,end):
        new_case_i = create_new_case(case300,0.8,1.0)
        #print(cur_total_netload)
        Hi,Yi,Si = case2AandH(new_case_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/testSet/input/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/testSet/input/casezj_Y_%s'%i, Yi)

        
        res_i = runpf.runpf(new_case_i)[0]
        if res_i['success'] != 1:
            print('not conv')
            break
        Hi,Yi,Si = case2AandH(res_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/testSet/output/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/testSet/output/casezj_Y_%s'%i, Yi)

        print("\r%s/%s"%(i-start,end-start),end='\r')


# In[]
if __name__ == "__main__":
    start = 1000
    end = 2000
    for i in range(start,end):
        new_case_i = create_new_case(case300,0.2,0.8)
        #print(cur_total_netload)
        Hi,Yi,Si = case2AandH(new_case_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/testSet/input/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/testSet/input/casezj_Y_%s'%i, Yi)

        
        res_i = runpf.runpf(new_case_i)[0]
        if res_i['success'] != 1:
            print('not conv')
            break
        Hi,Yi,Si = case2AandH(res_i)
        sparse_Yi = csr_matrix(Yi)
        np.save(path+r'/数据/潮流(图)格式/300-system/testSet/output/casezj_H_%s.npy'%i, Hi)
        save_npz(path+r'/数据/潮流(图)格式/300-system/testSet/output/casezj_Y_%s'%i, Yi)

        print("\r%s/%s"%(i-start,end-start),end='\r')

# %%

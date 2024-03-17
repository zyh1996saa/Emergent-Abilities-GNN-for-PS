# -*- coding: utf-8 -*-
"""
 pre-trained graph transformer 
"""

import numpy as np
import pandas as pd
import copy

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


ID_name_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='节点',index_col=0)

branch_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='支路',index_col=0)

outer_power_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='外来电',index_col=0)

wind_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='风电',index_col=0)

pv_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='光伏',index_col=0)

nc_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='核电',index_col=0)

load_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='负荷',index_col=0)

flex_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='灵活资源',index_col=0)

def PQ():
    isPQ = np.zeros((ori_case['bus'].shape[0],))
    wherePQ = np.where(ori_case['bus'][:,1]==1)[0]
    for i in range(wherePQ.shape[0]):
        isPQ[wherePQ[i]] = 1
    return isPQ

def PV():
    isPV = np.zeros((ori_case['bus'].shape[0],))
    wherePV = np.where(ori_case['bus'][:,1]==2)[0]
    for i in range(wherePV.shape[0]):
        isPV[wherePV[i]] = 1
    return isPV

def Pt():
    isPt= np.zeros((ori_case['bus'].shape[0],))
    wherePt = np.where(ori_case['bus'][:,1]==3)[0]
    for i in range(wherePt.shape[0]):
        isPt[wherePt[i]] = 1
    return isPt

def load_H_from_cases(i,sample_i):
    print("加载潮流文件中:")
    h_in = np.zeros((sample_i, h_in_shape0, 6))
    #a_in = np.zeros((sample_i, h_in_shape0,  h_in_shape0))
    for sample_num in range(i*sample_i,(i+1)*sample_i):
        print("\r进度%s/%s"%(sample_num,sample_i+i*sample_i),end='\r')
        file = np.load(path + r'\数据\潮流样本与结果\casezj_%s.npz'%sample_num)
        tempcase = {key: file[key] for key in file}
        gbus = (tempcase['gen'][:,0] - 1).astype('int')
        # h_in 的列向量依次为负载有功、负载无功、发电机有功、发电机无功、电压幅值、电压相角
        h_in[sample_num-i*sample_i,:,0] = tempcase['bus'][:,2]
        h_in[sample_num-i*sample_i,:,1] = tempcase['bus'][:,3]
        h_in[sample_num-i*sample_i,gbus,2] = tempcase['gen'][:,1]
        h_in[sample_num-i*sample_i,gbus,3] = tempcase['gen'][:,2]
        h_in[sample_num-i*sample_i,:,4] = tempcase['bus'][:,7]
        h_in[sample_num-i*sample_i,:,5] = tempcase['bus'][:,8]
    return h_in

def load_Hout_from_res(i,sample_i):
    print("加载潮流结果中:")
    
    h_out = np.zeros((sample_i, h_in_shape0, 6))
    #a_in = np.zeros((sample_i, h_in_shape0,  h_in_shape0))
    for sample_num in range(i*sample_i,(i+1)*sample_i):
        #print(sample_num)
        print("\r进度%s/%s"%(sample_num,sample_i+i*sample_i),end='\r')
        file = np.load(path+r'\数据\潮流样本与结果\res_%s.npz'%sample_num,allow_pickle=True)
        tempcase = {key: file[key] for key in file}
        gbus = (tempcase['gen'][:,0] - 1).astype('int')
        # h_in 的列向量依次为负载有功、负载无功、发电机有功、发电机无功、电压幅值、电压相角
        h_out[sample_num-i*sample_i,:,0] = tempcase['bus'][:,2]
        h_out[sample_num-i*sample_i,:,1] = tempcase['bus'][:,3]
        h_out[sample_num-i*sample_i,gbus,2] = tempcase['gen'][:,1]
        h_out[sample_num-i*sample_i,gbus,3] = tempcase['gen'][:,2]
        h_out[sample_num-i*sample_i,:,4] = tempcase['bus'][:,7]
        h_out[sample_num-i*sample_i,:,5] = tempcase['bus'][:,8]
    return h_out

def norm_H(H_in):
    # 对于每个节点的每种属性计算平均值和标准差
    mean_per_node = np.round(np.mean(H_in, axis=0),4)
    std_per_node = np.round(np.std(H_in, axis=0),4)
    
    # 创建一个与H_in形状相同的数组，用于存放标准化后的数据
    H_normalized_per_node = np.zeros_like(H_in)
    
    # 遍历每个节点的每种属性
    for i in range(H_in.shape[1]):  # 遍历所有节点
        for j in range(H_in.shape[2]):  # 遍历所有属性
            # 如果标准差不为零，则进行标准化
            if std_per_node[i, j] != 0:
                H_normalized_per_node[:, i, j] = (H_in[:, i, j] - mean_per_node[i, j]) / std_per_node[i, j]
            else:
                # 标准差为零时，只减去平均值
                H_normalized_per_node[:, i, j] = H_in[:, i, j] - mean_per_node[i, j]

    return H_normalized_per_node, mean_per_node, std_per_node

def rebuild_h(h,mean,std):
    return h*std+mean

def create_model():
    # 定义模型
    
    hin = tf.keras.Input(shape=(1581,6))
    
    p_load_in = hin[:,:,0]
    
    h_hidden = tf.keras.layers.Flatten(input_shape=(1581, 6))(hin)
    h_hidden = tf.keras.layers.Dense(1024,name='dense1',activation='gelu')(h_hidden)
    h_hidden = tf.keras.layers.Dense(1024,name='dense2',activation='gelu')(h_hidden)
    
    hout = tf.keras.layers.Dense(1581*6,name='dense3',activation='gelu')(h_hidden)
    
    hout_reshape = tf.reshape(hout, [-1, 1581,6])
    
    p_load_out = hout_reshape[:,:,0]
    loss1 = tf.reduce_mean(tf.math.abs(p_load_out * isPQ - p_load_in))
    
    
    model = tf.keras.Model([hin,], [hout])
    model.compile(optimizer='adam',
              loss='mean_squared_error',  # 假设是回归问题
              metrics=['mean_absolute_error'])
    
    loss = 2*loss1
    
    
    model.add_metric(loss1,name='loss1')
    
    model.add_loss(loss)
    
    return model


if __name__ == "__main__":
    
    path = r'F:\预训练大模型'
    
    ori_case = np.load(path+r'\数据\潮流样本与结果\res_0.npz',allow_pickle=True)
    ori_case = {key: ori_case[key] for key in ori_case}
    
    isPQ = PQ()
    isPV = PV()
    isPt = Pt()
    
    total_sample_num = 6000   
    sample_for_each_iter = 2000
    
    h_in_shape0 = ID_name_tab.shape[0]
    
    model = create_model()

    for i in range(int(total_sample_num/sample_for_each_iter)):
        
        H_in =  load_H_from_cases(i,sample_for_each_iter)
        H_in_norm,H_in_mean,H_in_std = norm_H(H_in)
        
        H_out = load_Hout_from_res(i,sample_for_each_iter)
        H_out_norm,H_out_mean,H_out_std = norm_H(H_out)
    
        model.fit(H_in_norm, tf.reshape(H_out_norm, [-1, 1581*6]), epochs=int(sample_for_each_iter/10), batch_size=int(sample_for_each_iter/10), validation_split=0.2,verbose=1)
        #model.fit(H_in_norm, tf.reshape(H_out_norm, [-1, 1581*6]), epochs=1, batch_size=int(sample_for_each_iter/10), validation_split=0.2,verbose=1)
    H_pre_norm = tf.reshape(model.predict(H_in_norm), [sample_for_each_iter, 1581,6]).numpy()
        
    H_pre = rebuild_h(H_pre_norm,H_out_mean,H_out_std)


# In[]


# 性能测试

if __name__ == "__main__":
    
    test_sample_num = 2000
    i = 4
    
    H_in_test =  load_H_from_cases(i,test_sample_num)
    H_in_norm_test,H_in_mean_test,H_in_std_test = norm_H(H_in_test)
    
    H_out_test = load_Hout_from_res(i,test_sample_num)
    H_out_norm_test,H_out_mean_test,H_out_std_test = norm_H(H_out_test)
    
    H_pre_norm = tf.reshape(model.predict(H_in_norm_test), [test_sample_num, 1581,6]).numpy()
    H_pre = rebuild_h(H_pre_norm,H_out_mean_test,H_out_std_test)
    
    model.evaluate(H_in_norm_test,tf.reshape(H_out_norm_test, [-1, 1581*6]))
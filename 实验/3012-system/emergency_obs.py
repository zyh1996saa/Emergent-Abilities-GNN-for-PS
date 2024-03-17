# In[]
import sys
sys.path.append(r'../')
sys.path.append(r'../潮流')
import numpy as np
import pandas as pd
import copy
import pypower.runpf as runpf
sys.path.append(r'../systemFiles')

import case3012 
ori_case = case3012.load_case_data()
sys_size = ori_case['bus'].shape[0]

print("%s-system 加载成功"%sys_size)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from scipy.sparse import load_npz
from gat_float import GraphAttentionLayer as GAT
from GCN import GCNLayer
from tensorflow.keras.callbacks import EarlyStopping



path = r'/home/user/Desktop/预训练大模型'


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

def load_H(start_label,end_label,dataset='trainingSet',datatype='input',sample_for_each_iter=1000):
    filepath = path + r'/数据/潮流(图)格式/%s-system/%s/%s'%(sys_size,dataset,datatype)
    H_in = np.zeros((end_label-start_label,h_in_shape0,6))
    for _ in range(start_label,end_label):
        H_in[_%sample_for_each_iter,:,:] = np.load(filepath+r'/casezj_H_%s.npy'%_)
        print('\r加载进度%s/%s'%(_,end_label-start_label),end='\r')
    return H_in

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

def norm_H_given_Mean_Std(H,H_mean,H_std):
    H_normalized_per_node = np.zeros_like(H)
    for i in range(H.shape[1]):  # 遍历所有节点
        for j in range(H.shape[2]):  # 遍历所有属性
            # 如果标准差不为零，则进行标准化
            if H_std[i, j] != 0:
                H_normalized_per_node[:, i, j] = (H[:, i, j] - H_mean[i, j]) / H_std[i, j]
            else:
                # 标准差为零时，只减去平均值
                H_normalized_per_node[:, i, j] = H[:, i, j] - H_mean[i, j]

    return H_normalized_per_node

def create_mlp(pl=[64,64],mlp_depth=2):
    hin = tf.keras.Input(shape=(h_in_shape0,6),dtype=tf.float32,name='input')
    h_mlp_in = tf.keras.layers.Reshape((sys_size*6,))(hin)
    h_mlp_hidden = tf.keras.layers.Dense(pl[0], activation='relu',name='dense7')(h_mlp_in)
    if mlp_depth >= 2:
        h_mlp_hidden = tf.keras.layers.Dense(pl[1], activation='relu',name='dense8')(h_mlp_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(sys_size*6, name='dense9')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.Reshape((sys_size,6))(h_mlp_hidden)
    hout = h_mlp_out

    #p_load_out = tf.math.real(hout[:,:,0])
    
    #loss1 = tf.reduce_mean(tf.math.abs(p_load_out * isPQ - p_load_in))
    
    
    model = tf.keras.Model([hin], [hout])
    model.compile(optimizer='adam',
              loss='MSE',  # 假设是回归问题
              #metrics=['mean_absolute_error']
              )
    return model

def create_GCN(pl=[8,8],mlp_depth=2):
    # 定义模型
    hin = tf.keras.Input(shape=(h_in_shape0,6),dtype=tf.float32,name='input')
    
    #p_load_in = tf.math.real(hin[:,:,0])
    
    h_hidden = GCNLayer(units=1024, activation='relu',name='GCN1')([hin,A])   
    h_hidden = GCNLayer(units=6, name='GCN2')([h_hidden,A])   
    #h_hidden = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm1')(h_hidden + hin)

    #h_hidden = GAT(units=6, num_heads=8)([h_hidden,A])
    #h_mlp_in = tf.keras.layers.Dense(64, activation='relu',name='dense1')(h_hidden)
    #h_mlp_hidden = tf.keras.layers.Dense(16, activation='relu',name='dense2')(h_mlp_in)
    #h_mlp_out = tf.keras.layers.Dense(6,name='dense3')(h_mlp_hidden)
    #h_mlp_out = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm2')(h_mlp_out + hin)
    
    h_mlp_in = tf.keras.layers.Reshape((sys_size*6,))(h_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(pl[0], activation='relu',name='dense7')(h_mlp_in)
    if mlp_depth>=2:
        h_mlp_hidden = tf.keras.layers.Dense(pl[1], activation='relu',name='dense8')(h_mlp_hidden)
    if mlp_depth>=3:
        h_mlp_hidden = tf.keras.layers.Dense(pl[2], activation='relu',name='dense10')(h_mlp_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(sys_size*6, name='dense9')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.Reshape((sys_size,6))(h_mlp_hidden)

    hout = h_mlp_out

    #p_load_out = tf.math.real(hout[:,:,0])
    
    #loss1 = tf.reduce_mean(tf.math.abs(p_load_out * isPQ - p_load_in))
    
    
    model = tf.keras.Model([hin], [hout])
    model.compile(optimizer='adam',
              loss='MSE',  # 假设是回归问题
              #metrics=['mean_absolute_error']
              )
    
    #loss = 2*loss1
    
    
    #model.add_metric(loss1,name='loss1')
    
    #model.add_loss(loss)
    
    return model


def create_model(parameter_list=[8,8,16,9,6,8,8,16,9,6],mlp_depth=2):
    # 定义模型
    pl = parameter_list
    hin = tf.keras.Input(shape=(h_in_shape0,6),dtype=tf.float32,name='input')
    
    #p_load_in = tf.math.real(hin[:,:,0])
    
    h_hidden = GAT(units=6, num_heads=8,name='GAT1')([hin,A])   
    h_hidden = GAT(units=6, num_heads=8, output_layer=True,name='GAT2')([h_hidden,A])   
    h_hidden = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm1')(h_hidden + hin)
    tempstor_1 = h_hidden
    #h_hidden = GAT(units=6, num_heads=8)([h_hidden,A])
    h_mlp_in = tf.keras.layers.Dense(64, activation='relu',name='dense1')(h_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(16, activation='relu',name='dense2')(h_mlp_in)
    h_mlp_out = tf.keras.layers.Dense(6,name='dense3')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm2')(h_mlp_out + hin)

    
    tempstor_2 = h_mlp_out

    h_hidden = GAT(units=6, num_heads=8,name='GAT3')([h_mlp_out,A])   
    h_hidden = GAT(units=6, num_heads=8, output_layer=True,name='GAT4')([h_hidden,A]) 
    h_hidden = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm3')(h_hidden + hin)
    #h_hidden = GAT(units=6, num_heads=8)([h_hidden,A])
    h_mlp_in = tf.keras.layers.Dense(512, activation='relu',name='dense4')(h_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(64, activation='relu',name='dense5')(h_mlp_in)
    h_mlp_out = tf.keras.layers.Dense(6, name='dense6')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm4')(h_mlp_out + hin)
    
    h_mlp_in = tf.keras.layers.Reshape((sys_size*6,))(h_mlp_out)
    h_mlp_hidden = tf.keras.layers.Dense(pl[0], activation='relu',name='dense7')(h_mlp_in)
    if mlp_depth>=2:
        h_mlp_hidden = tf.keras.layers.Dense(pl[1], activation='relu',name='dense8')(h_mlp_hidden)
    if mlp_depth>=3:
        h_mlp_hidden = tf.keras.layers.Dense(pl[2], activation='relu',name='dense10')(h_mlp_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(sys_size*6, name='dense9')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.Reshape((sys_size,6))(h_mlp_hidden)

    hout = h_mlp_out

    #p_load_out = tf.math.real(hout[:,:,0])
    
    #loss1 = tf.reduce_mean(tf.math.abs(p_load_out * isPQ - p_load_in))
    
    
    model = tf.keras.Model([hin], [hout])
    model.compile(optimizer='adam',
              loss='MSE',  # 假设是回归问题
              #metrics=['mean_absolute_error']
              )
    
    #loss = 2*loss1
    
    
    #model.add_metric(loss1,name='loss1')
    
    #model.add_loss(loss)
    
    return model

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

# 定义 EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # 监视验证集上的损失
    min_delta=0.0002,  # 被认为是改进的最小变化
    patience=30,  # 在停止之前，可以容忍没有改进的轮数
    verbose=1,  # 打印停止信息
    mode='min',  # 寻找最小值
    restore_best_weights=True  # 恢复最佳模型的权重
)

if __name__ == "__main__":
    
    """
    加载前置文件
    """


    ori_case['version'] = '2'
    ori_case = refresh_busnum(ori_case)
    ori_case_res = runpf.runpf(ori_case)
    

    isPQ = PQ()
    isPV = PV()
    isPt = Pt()



    h_in_shape0 = ori_case['bus'].shape[0]
    
    # 加载稀疏矩阵并转换为密集形式
    Y = load_npz(path + r'/数据/潮流(图)格式/%s-system/trainingSet/input/casezj_Y_0.npz'%sys_size).toarray()
    #Y = load_npz(path + r'/数据/潮流(图)格式/%s-system/trainingSet/input/casezj_H_0.npy.npz'%sys_size).toarray()
    # 转换为 TensorFlow 张量并创建布尔掩码
    A = tf.cast(np.where(Y != 0, 1, 0), dtype=tf.float32)

    model = create_GCN()
    PL = [
        [1,1,1],
        #[1,2],
        [2,2,2],
        #[1,3],
        #[1,4],
        #3,2],
        #[3,3],
        #[3,4],
        #[3,4],
        [4,4,4],
        [8,8,8],
        [16,16,16],
        #[32,16],
        [32,32,32],
        #[64,32],
        [64,64,64],
        #[128,64],
        [128,128,128],
        [256,256,256],
        [512,512,512],
        [1024,1024,1024],
        [2048,2048,2048],
        #[4096,2048],
    ]

    model_types = ['GCN','GAT',"mlp"]
    model_type = 'GCN'
# In[]
if __name__ == "__main__":


    total_sample_num = 5000  
    sample_for_each_iter = 5000

    H_in =  load_H(0,total_sample_num,sample_for_each_iter=total_sample_num)   
    H_in_norm,H_in_mean_train,H_in_std_train = norm_H(H_in)

    H_out = load_H(0,total_sample_num,'trainingSet','output',sample_for_each_iter=total_sample_num)      
    H_out_norm_train,H_out_mean_train,H_out_std_train = norm_H(H_out)

    total_sample_num = 1000  
    sample_for_each_iter = 1000
    i = 0
    dataset = "testSet"

    H_in_test =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'input',sample_for_each_iter=sample_for_each_iter) 
    H_in_norm_test = norm_H_given_Mean_Std(H_in_test,H_in_mean_train,H_in_std_train)

    H_out_test = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'output',sample_for_each_iter=sample_for_each_iter)
    H_out_norm_test = norm_H_given_Mean_Std(H_out_test,H_out_mean_train,H_out_std_train)
    
    for pl in PL:
        mlp_depth = len(pl)
        #model = create_model(pl,mlp_depth=mlp_depth)
        #model = create_mlp(pl,mlp_depth=mlp_depth)
        if model_type == "GCN":
            model = create_GCN(pl,mlp_depth=mlp_depth)
        if model_type == "mlp":
            model = create_mlp(pl,mlp_depth=mlp_depth)
        

        for i in range(int(total_sample_num/sample_for_each_iter)):
            
            print("正加载第%s批数据"%i)

            H_in =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,sample_for_each_iter=sample_for_each_iter)      
            H_in_norm = norm_H_given_Mean_Std(H_in,H_in_mean_train,H_in_std_train)
            
            H_out = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,'trainingSet','output',sample_for_each_iter=sample_for_each_iter)
            H_out_norm = norm_H_given_Mean_Std(H_out,H_out_mean_train,H_out_std_train)
            print("training neural network with parameters %s, pl=%s " %(model.count_params(),pl))
            #model.fit(H_in_norm, H_out_norm, epochs=int(sample_for_each_iter/10), batch_size=int(sample_for_each_iter/100), validation_split=0.2,verbose=1)
            model.fit(H_in_norm, H_out_norm, epochs=1000, batch_size=int(sample_for_each_iter/100), validation_split=0.2,verbose=1,callbacks=[early_stopping])

        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()  # 获取层的参数
            para_num = model.count_params()
            np.savez(path+r'/saved_models/%s-system/%s_weights_%s'%(sys_size,model_type,para_num)+f'layer_{i}_weights.npz', *weights)
        print("param num: %s, pl = %s"%(para_num,pl))
        model.evaluate(H_in_norm_test,H_out_norm_test)

# %%
import matplotlib.pyplot as plt
if __name__ == "__main__":

    total_sample_num = 1000  
    sample_for_each_iter = 1000

    H_in =  load_H(0,total_sample_num,sample_for_each_iter=total_sample_num)   
    H_in_norm,H_in_mean_train,H_in_std_train = norm_H(H_in)

    H_out = load_H(0,total_sample_num,'trainingSet','output',sample_for_each_iter=total_sample_num)      
    H_out_norm_train,H_out_mean_train,H_out_std_train = norm_H(H_out)

    para_nums = []
    scores = []

    total_sample_num = 1000  
    sample_for_each_iter = 1000
    i = 0
    dataset = "testSet"

    H_in =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'input',sample_for_each_iter=sample_for_each_iter) 
    H_in_norm = norm_H_given_Mean_Std(H_in,H_in_mean_train,H_in_std_train)

    H_out = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'output',sample_for_each_iter=sample_for_each_iter)
    H_out_norm = norm_H_given_Mean_Std(H_out,H_out_mean_train,H_out_std_train)

    H_out_predict_norm = model.predict(H_in_norm)
    H_out_predict = H_out_predict_norm * H_out_std_train + H_out_mean_train
    
    for pl in PL:
        
        mlp_depth = len(pl)
        #model = create_mlp(pl,mlp_depth=mlp_depth)
        #model = create_mlp(pl,mlp_depth=mlp_depth)
        if model_type == "GCN":
            model = create_GCN(pl,mlp_depth=mlp_depth)
        if model_type == "mlp":
            model = create_mlp(pl,mlp_depth=mlp_depth)
        para_num = model.count_params()

        load_success_flag = 1

        for i, layer in enumerate(model.layers):
            try:
                if model_type == "GCN":
                    weights = np.load(path+r'/saved_models/%s-system/%s_weights_%s'%(sys_size,model_type,para_num)+f'layer_{i}_weights.npz')
                    layer.set_weights([weights[f'arr_{j}'] for j in range(len(weights))])
                if model_type == "mlp":
                    weights = np.load(path+r'/saved_models/%s-system/GAT_noPI_weights_%s'%(sys_size,para_num)+f'layer_{i}_weights.npz')
                    layer.set_weights([weights[f'arr_{j}'] for j in range(len(weights))])
            except:
                load_success_flag = 0
                break


        if not load_success_flag:
            continue
        
        para_nums.append(para_num)
        

        print(pl)
        score = model.evaluate(H_in_norm,H_out_norm)
        
        scores.append(score)

    plt.scatter(para_nums, scores)
    plt.xscale('log')
    plt.show()

    
# %%
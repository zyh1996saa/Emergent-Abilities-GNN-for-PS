# In[]
import numpy as np
import pandas as pd
import copy

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from scipy.sparse import load_npz
from gat_float import GraphAttentionLayer as GAT

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

def load_H(start_label,end_label,dataset='训练集',datatype='输入',sample_for_each_iter=1000):
    filepath = path + r'/数据/潮流(图)格式/1581节点系统/%s/%s'%(dataset,datatype)
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


def create_model(isPI=True):
    # 定义模型
    
    hin = tf.keras.Input(shape=(1581,6),dtype=tf.float32,name='input')

    hin_rebuild = hin * H_in_std_train + H_in_mean_train
    
    p_load_in = hin_rebuild[:,:,0]
    q_load_in = hin_rebuild[:,:,1]
    p_gen_in = hin_rebuild[:,:,2]
    v_gen_in = hin_rebuild[:,:,4]
    q_gen_in = hin_rebuild[:,:,3]
    theta_gen_in = hin_rebuild[:,:,5]
    
    
    h_hidden = GAT(units=6, num_heads=8,name='GAT1')([hin,A])   
    h_hidden = GAT(units=6, num_heads=8, output_layer=True,name='GAT2')([h_hidden,A])   
    h_hidden = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm1')(h_hidden + hin)

    h_mlp_in = tf.keras.layers.Dense(16, activation='relu',name='dense1')(h_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(9, activation='relu',name='dense2')(h_mlp_in)
    h_mlp_out = tf.keras.layers.Dense(6, activation='relu',name='dense3')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm2')(h_mlp_out + h_hidden)

    h_hidden = GAT(units=6, num_heads=8,name='GAT3')([h_mlp_out,A])   
    h_hidden = GAT(units=6, num_heads=8, output_layer=True,name='GAT4')([h_hidden,A]) 
    h_hidden = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm3')(h_hidden + hin)

    h_mlp_in = tf.keras.layers.Dense(16, activation='relu',name='dense4')(h_hidden)
    h_mlp_hidden = tf.keras.layers.Dense(9, activation='relu',name='dense5')(h_mlp_in)
    h_mlp_out = tf.keras.layers.Dense(6, activation='relu',name='dense6')(h_mlp_hidden)
    h_mlp_out = tf.keras.layers.LayerNormalization(epsilon=1e-6,name='norm4')(h_mlp_out + h_hidden)

    
    hout = h_mlp_out

    hout_rebuild = hout * H_out_std_train + H_out_mean_train

    p_load_out = hout_rebuild[:,:,0]
    q_load_out = hout_rebuild[:,:,1]
    p_gen_out = hout_rebuild[:,:,2]
    v_gen_out = hout_rebuild[:,:,4]
    q_gen_out = hout_rebuild[:,:,3]
    theta_gen_out = hout_rebuild[:,:,5]
    
    loss1 = tf.reduce_mean(tf.math.abs(p_load_out * isPQ - p_load_in * isPQ)/H_out_mean_train[:,0].mean())
    loss2 = tf.reduce_mean(tf.math.abs(q_load_out * isPQ - q_load_in * isPQ)/H_out_mean_train[:,1].mean())
    loss3 = tf.reduce_mean(tf.math.abs(p_gen_out * isPV - p_gen_in * isPV)/H_out_mean_train[:,2].mean())
    loss4 = tf.reduce_mean(tf.math.abs(v_gen_out * isPV - v_gen_in * isPV)/H_out_mean_train[:,4].mean())
    loss5 = tf.reduce_mean(tf.math.abs(q_gen_out * isPt - q_gen_in * isPt)/H_out_mean_train[:,3].mean())
    loss6 = tf.reduce_mean(tf.math.abs(theta_gen_out * isPt - theta_gen_in * isPt)/H_out_mean_train[:,5].mean())

    p_out = p_gen_out - p_load_out
    q_out = q_gen_out - q_load_out
    theta_rad = tf.convert_to_tensor(theta_gen_out * (np.pi / 180.0), dtype=tf.float32)

    # 创建复数表示的电压
    V_complex = tf.complex(v_gen_out, tf.zeros_like(v_gen_out)) * tf.exp(tf.complex(tf.zeros_like(theta_rad), theta_rad))
    V_complex = tf.expand_dims(V_complex, axis=-1)
    Y_tensor = tf.convert_to_tensor(Y, dtype=tf.complex64)
    #print(V_complex.shape)
    #print(Y_tensor.shape)
    I = tf.linalg.matmul(Y_tensor, V_complex)
    I = tf.squeeze(I, axis=-1)
    #print(I.shape)
    V_complex = tf.squeeze(V_complex, axis=-1)
    S = V_complex * tf.math.conj(I)
    P_calculated = tf.math.real(S)
    Q_calculated = tf.math.imag(S)
    loss7 = tf.reduce_mean(tf.math.abs(baseMVA*P_calculated-p_out)/H_out_mean_train[:,0].mean())
    loss8 = tf.reduce_mean(tf.math.abs(baseMVA*Q_calculated-q_out)/H_out_mean_train[:,1].mean())
    
    model = tf.keras.Model([hin], [hout])
    model.compile(optimizer='adam',
              loss='MSE',  
              metrics=['mean_absolute_error']
              )
    
    loss = 2*loss1 + 2*loss2 + 2*loss3 + 2*loss4 + 2*loss5 + 2*loss6 + 2*loss7 + 2*loss8
    
    if isPI:
        model.add_metric(loss1,name='deltaP_of_PQ')
        model.add_metric(loss2,name='deltaQ_of_PQ')
        model.add_metric(loss3,name='deltaP_of_PV')
        model.add_metric(loss4,name='deltaV_of_PV')
        model.add_metric(loss5,name='deltaP_of_Pt')
        model.add_metric(loss6,name='deltat_of_Pt')
        model.add_metric(loss7,name='deltaP_PFE')
        model.add_metric(loss8,name='deltaQ_PFE')
        
        model.add_loss(loss)
    
    return model

if __name__ == "__main__":
    
    """
    加载前置文件
    """
    ID_name_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='节点',index_col=0)
    
    branch_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='支路',index_col=0)
    
    outer_power_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='外来电',index_col=0)
    
    wind_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='风电',index_col=0)
    
    pv_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='光伏',index_col=0)
    
    nc_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='核电',index_col=0)
    
    load_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='负荷',index_col=0)
    
    flex_tab = pd.read_excel('flex_assessment_configs.xlsx', sheet_name='灵活资源',index_col=0)
    
    
    ori_case = {key: np.load(path+r'/实验/潮流/zj2025.npz')[key] for key in np.load('zj2025.npz')}
    
    isPQ = PQ()
    isPV = PV()
    isPt = Pt()



    h_in_shape0 = ID_name_tab.shape[0]
    
    # 加载稀疏矩阵并转换为密集形式
    Y = load_npz(path + r'/数据/潮流(图)格式/1581节点系统/测试集/输入/casezj_Y_0.npz').toarray()

    # 转换为 TensorFlow 张量并创建布尔掩码
    A = tf.cast(np.where(Y != 0, 1, 0), dtype=tf.float32)

    baseMVA = 100
        
    total_sample_num = 20000 
    sample_for_each_iter = 200

    H_in =  load_H(0,total_sample_num,sample_for_each_iter=total_sample_num)   
    H_in_norm,H_in_mean_train,H_in_std_train = norm_H(H_in)

    H_out = load_H(0,total_sample_num,'训练集','输出',sample_for_each_iter=total_sample_num)      
    H_out_norm_train,H_out_mean_train,H_out_std_train = norm_H(H_out)

    model = create_model()
# In[]
if __name__ == "__main__":
    for i in range(int(total_sample_num/sample_for_each_iter)):
        
        print("正加载第%s批数据"%i)

        H_in =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,sample_for_each_iter=sample_for_each_iter)      
        H_in_norm = norm_H_given_Mean_Std(H_in,H_in_mean_train,H_in_std_train)
        
        H_out = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,'训练集','输出',sample_for_each_iter=sample_for_each_iter)
        H_out_norm = norm_H_given_Mean_Std(H_out,H_out_mean_train,H_out_std_train)

        #H_in_norm,H_out_norm = tf.cast(H_in_norm,dtype='float16'),tf.cast(H_out_norm,dtype='float16')
        model.fit(H_in_norm, H_out_norm, epochs=int(sample_for_each_iter/10), batch_size=int(sample_for_each_iter/100), validation_split=0.2,verbose=1)

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()  # 获取层的参数
        np.savez(path+r'/saved_models/1581-sys/GAT_withPI_weights/'+f'layer_{i}_weights.npz', *weights)


# %%
"""
if __name__ == "__main__":

    model = tf.saved_model.load(path+r'/saved_models/GAT_noPI')
    infer = model.signatures["serving_default"]


    total_sample_num = 1000  
    sample_for_each_iter = 200
    i = 4
    dataset = "测试集"

    H_in =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'输入',sample_for_each_iter=sample_for_each_iter) 
    H_in_norm,H_in_mean,H_in_std = norm_H(H_in)
    H_in_norm = {'input': tf.constant(H_in_norm[:,:,:],dtype=tf.float32)}
        
    H_out = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'输出',sample_for_each_iter=sample_for_each_iter)
    #H_out_norm,H_out_mean,H_out_std = norm_H(H_out)
    H_out_norm = norm_H_given_Mean_Std(H_out,H_out_mean_train,H_out_std_train)

    H_out_predict_norm = infer(**H_in_norm)['norm4']
    H_out_predict = H_out_predict_norm * H_out_std_train + H_out_mean_train

    print(tf.keras.losses.MeanSquaredError()(H_out_norm,H_out_predict_norm).numpy())
"""

# %%
if __name__ == "__main__":
    model = create_model(isPI=True)
    for i, layer in enumerate(model.layers):
        weights = np.load(path+r'/saved_models/1581-sys/GAT_withPI_weights/'+f'layer_{i}_weights.npz')
        layer.set_weights([weights[f'arr_{j}'] for j in range(len(weights))])
    
    total_sample_num = 1000  
    sample_for_each_iter = 1
    i = 2
    dataset = "训练集"

    H_in =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'输入',sample_for_each_iter=sample_for_each_iter) 
    H_in_norm = norm_H_given_Mean_Std(H_in,H_in_mean_train,H_in_std_train)

    H_out = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'输出',sample_for_each_iter=sample_for_each_iter)
    H_out_norm = norm_H_given_Mean_Std(H_out,H_out_mean_train,H_out_std_train)

    H_out_predict_norm = model.predict(H_in_norm)
    H_out_predict = H_out_predict_norm * H_out_std_train + H_out_mean_train

    model.evaluate(H_in_norm,H_out_norm)
# %%

total_sample_num = 1000  
sample_for_each_iter = 1
i = 2
dataset = "测试集"

H_in =  load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'输入',sample_for_each_iter=sample_for_each_iter) 
H_in_norm = norm_H_given_Mean_Std(H_in,H_in_mean_train,H_in_std_train)

H_out = load_H(i*sample_for_each_iter,(i+1)*sample_for_each_iter,dataset,'输出',sample_for_each_iter=sample_for_each_iter)
H_out_norm = norm_H_given_Mean_Std(H_out,H_out_mean_train,H_out_std_train)

H_in_rebuild = H_in_norm * H_in_std_train + H_in_mean_train
H_out_rebuild = H_out_norm * H_out_std_train + H_out_mean_train

loss1_test = H_in_rebuild[:,:,0] * isPQ - H_out_rebuild[:,:,0] * isPQ
loss2_test = H_in_rebuild[:,:,1] * isPQ - H_out_rebuild[:,:,1] * isPQ
loss3_test = H_in_rebuild[:,:,2] * isPV - H_out_rebuild[:,:,2] * isPV
loss4_test = H_in_rebuild[:,:,4] * isPV - H_out_rebuild[:,:,4] * isPV
loss5_test = H_in_rebuild[:,:,4] * isPt - H_out_rebuild[:,:,4] * isPt
loss6_test = H_in_rebuild[:,:,5] * isPt - H_out_rebuild[:,:,5] * isPt
print(np.all(loss1_test == 0),np.all(loss2_test == 0),np.all(loss3_test == 0),np.all(loss4_test == 0),np.all(loss5_test == 0),np.all(loss6_test == 0))

p_load_out = H_out_rebuild[:,:,0]
q_load_out = H_out_rebuild[:,:,1]
p_gen_out = H_out_rebuild[:,:,2]
q_gen_out = H_out_rebuild[:,:,3]
v_gen_out = H_out_rebuild[:,:,4]
theta_gen_out = H_out_rebuild[:,:,5]

p_out = p_gen_out - p_load_out
q_out = q_gen_out - q_load_out
theta_rad = np.radians(theta_gen_out)
V_complex = (v_gen_out * np.exp(1j * theta_rad)).reshape(1581,1)
I = np.dot(Y, V_complex)
S = V_complex * np.conj(I)
P_calculated = S.real
Q_calculated = S.imag
loss7_test = baseMVA*P_calculated.reshape(1581,) - p_out.reshape(1581,)
loss8_test = baseMVA*Q_calculated.reshape(1581,) - q_out.reshape(1581,)
# %%
H_in_rebuild = H_in_norm * H_in_std_train + H_in_mean_train
H_out_rebuild = H_out_predict_norm * H_out_std_train + H_out_mean_train
loss1_test = H_in_rebuild[:,:,0] * isPQ - H_out_rebuild[:,:,0] * isPQ
loss2_test = H_in_rebuild[:,:,1] * isPQ - H_out_rebuild[:,:,1] * isPQ
loss3_test = H_in_rebuild[:,:,2] * isPV - H_out_rebuild[:,:,2] * isPV
loss4_test = H_in_rebuild[:,:,4] * isPV - H_out_rebuild[:,:,4] * isPV
loss5_test = H_in_rebuild[:,:,4] * isPt - H_out_rebuild[:,:,4] * isPt
loss6_test = H_in_rebuild[:,:,5] * isPt - H_out_rebuild[:,:,5] * isPt
print(np.all(loss1_test == 0),np.all(loss2_test == 0),np.all(loss3_test == 0),np.all(loss4_test == 0),np.all(loss5_test == 0),np.all(loss6_test == 0))
# %%

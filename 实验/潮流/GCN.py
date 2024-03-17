# In[]
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
class GCNLayer(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_shape = input_shape[0][-1]
        self.kernel = self.add_weight(name='kernel', shape=(feature_shape, self.units), initializer='glorot_uniform', trainable=True)
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        features, adjacency = inputs
        
        # 为了简化，假设A是静态的且不随批次变化，我们需要将它扩展到每个批次
        batch_size = tf.shape(features)[0]
        A_expanded = tf.expand_dims(adjacency, 0)
        A_expanded = tf.tile(A_expanded, [batch_size, 1, 1])
        
        # 节点特征变换
        features_transformed = K.dot(features, self.kernel)
        
        # 邻居特征聚合，使用批量矩阵乘法
        features_aggregated = tf.matmul(A_expanded, features_transformed)
        
        # 应用激活函数
        if self.activation is not None:
            features_aggregated = self.activation(features_aggregated)
        
        return features_aggregated
    
if __name__ == "__main__":
    # 假设的输入数据
    num_nodes = 8387  # 节点数量
    num_features = 6  # 特征维度
    num_heads = 3    # 注意力头数量
    units = 6        # 注意力单元数
    
    H = np.random.rand(100, num_nodes, num_features)
    A = np.random.rand( num_nodes, num_nodes)
    
    # 创建图注意力层
    GCN = GCNLayer(units=units, )
    output = GCN([H, A])
    
    #print(output)
# %%

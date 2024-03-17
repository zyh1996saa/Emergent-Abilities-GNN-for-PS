import tensorflow as tf
import numpy as np

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.identity, l2=0.0):
        super(Attention, self).__init__()
        self.l2 = l2
        self.activation = activation
        self.units = units

    def build(self, input_shape):
        H_shape, A_shape = input_shape

        # 初始化权重为浮点数类型
        self.W = self.add_weight(
            shape=(H_shape[-1], self.units),
            initializer='glorot_uniform',
            dtype=tf.float32,
        )

        self.a_1 = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            dtype=tf.float32,
        )

        self.a_2 = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            dtype=tf.float32,
        )

    def call(self, inputs):
        H, A = inputs

        # 将权重转换为复数类型进行计算
        W_complex = tf.cast(self.W, dtype=tf.complex128)
        a_1_complex = tf.cast(self.a_1, dtype=tf.complex128)
        a_2_complex = tf.cast(self.a_2, dtype=tf.complex128)

        X = tf.matmul(H, W_complex)

        attn_self = tf.matmul(X, a_1_complex)
        attn_neighbours = tf.matmul(X, a_2_complex)

        attention = attn_self + tf.transpose(attn_neighbours, perm=[0, 2, 1])

        E1 = tf.nn.leaky_relu(tf.math.real(attention))
        E2 = tf.nn.leaky_relu(tf.math.imag(attention))
        E = tf.complex(E1, E2)

        mask = -10e9 * (1.0 - A)
        masked_E = E + mask

        alpha1 = tf.nn.softmax(tf.math.real(masked_E))
        alpha2 = tf.nn.softmax(tf.math.imag(masked_E))
        alpha = tf.complex(alpha1, alpha2)

        H_cap = alpha @ X
        out1 = self.activation(tf.math.real(H_cap))
        out2 = self.activation(tf.math.imag(H_cap))
        out = tf.complex(out1, out2)

        return out




class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, output_layer=False, activation=tf.identity, l2=0.0):
        super(GraphAttentionLayer, self).__init__()
        self.activation = activation
        self.num_heads = num_heads
        self.output_layer = output_layer
        self.attn_layers = [Attention(units, l2=l2) for _ in range(num_heads)]

    def call(self, inputs):
        H, A = inputs
        H_out = [self.attn_layers[i]([H, A]) for i in range(self.num_heads)]

        if self.output_layer:
            multi_head_attn = tf.reduce_mean(tf.stack(H_out), axis=0)
        else:
            multi_head_attn = tf.concat(H_out, axis=-1)

        out1 = self.activation(tf.math.real(multi_head_attn))
        out2 = self.activation(tf.math.imag(multi_head_attn))
        out = tf.complex(out1, out2)

        return out

if __name__ == "__main__":
    # 假设的输入数据
    num_nodes = 10  # 节点数量
    num_features = 4  # 特征维度
    num_heads = 3    # 注意力头数量
    units = 6        # 注意力单元数
    
    H = tf.complex(np.random.rand(1, num_nodes, num_features), np.random.rand(1, num_nodes, num_features))
    A = tf.complex(np.random.rand(1, num_nodes, num_nodes), np.random.rand(1, num_nodes, num_nodes))
    
    # 创建图注意力层
    gat_layer = GraphAttentionLayer(units=units, num_heads=num_heads)
    output = gat_layer([H, A])
    
    print(output)
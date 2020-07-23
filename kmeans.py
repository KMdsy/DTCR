'''
Created on 2020年7月14日

@author: Shaoyu Dou
'''

import tensorflow as tf

class kmeans():
    def __init__(self, opts):
        self.hidden_dim = opts['encoder_hidden_units'][-1]
        self.training_samples_num = opts['training_samples_num']
        self.cluster_num = opts['cluster_num']
        self.F = tf.Variable(tf.random_normal(shape=(self.training_samples_num, self.cluster_num),mean=0,stddev=0.1), name='kmeans_F', dtype=tf.float32)
        
        
    def kmeans_optimalize(self, h):
        self.H = tf.transpose(h, [1,0], name='kmeans_H') # shape: hidden_dim, training_samples_num, m*N
        loss_kmeans = tf.subtract(tf.trace(tf.matmul(self.H, self.H, transpose_a=True)), 
                                  tf.trace(tf.matmul(tf.matmul(tf.matmul(self.F, self.H, transpose_a=True, transpose_b=True), self.H), self.F)), 
                                  name='l_kmeans') 
        return loss_kmeans
    
    def update_f(self, new_value):
        self.F = tf.assign(self.F, new_value)
        return self.F #tensor
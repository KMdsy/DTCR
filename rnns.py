'''
Created on 2020年7月14日

@author: Shaoyu Dou
'''
import tensorflow as tf
import drnn
from tensorflow.python.ops import array_ops

class dilated_encoder():
    def __init__(self, opts):
        self.hidden_units = opts['encoder_hidden_units']
        self.dilations = opts['dilations']
        assert(len(self.hidden_units) == len(self.dilations))
        
        
    def encoder(self, inputs):
        # 正向输入
        cell_fw_list = [tf.nn.rnn_cell.GRUCell(num_units=units) for units in self.hidden_units]
        #state_fw.shape = [batchsize, units], ..., [batchsize, units]
        outputs_fw, states_fw = drnn.multi_dRNN_with_dilations(cell_fw_list, inputs, self.dilations, scope='forward_drnn')

        # 逆向输入
        batch_axis = 0
        time_axis = 1
        inputs_bw = array_ops.reverse(inputs, axis=[time_axis])

        cell_bw_list = [tf.nn.rnn_cell.GRUCell(num_units=units) for units in self.hidden_units]
        outputs_bw, states_bw = drnn.multi_dRNN_with_dilations(cell_bw_list, inputs_bw, self.dilations, scope='backward_drnn')        
        outputs_bw = array_ops.reverse(outputs_bw, axis=[time_axis])# 与输出相对

        #输出
        states_fw = tf.concat(states_fw, axis=1)# [batchsize, units1 + units2 + units3]
        states_bw = tf.concat(states_bw, axis=1)# [batchsize, units1 + units2 + units3]
        final_states = tf.concat([states_fw, states_bw], axis=1)# [batchsize, 2*(units1 + units2 + units3)]        
        
        return final_states
    
class single_layer_decoder():
    def __init__(self, opts):
        self.hidden_units = 2 * sum(opts['encoder_hidden_units'])
        
    def decoder(self, init_state, init_input):
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_units)
        
        # 在原版的GRU中，ht = f(xt, ht-1), 即原本下一时刻都使用上一时刻的输出，当decoder中输入了有效的xt后，即使用了上一时刻的输出与本时刻的输入
        
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=init_input, initial_state=init_state)
        
        recons = outputs[:, :, 0]
        recons = tf.expand_dims(recons, axis=2)
        
        return recons
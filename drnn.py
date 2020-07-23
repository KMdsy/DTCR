'''
Created on 2020年7月14日

@author: Shaoyu Dou
'''
import copy
import itertools
import numpy as np
import tensorflow as tf

def dRNN(cell, inputs, rate, scope='default'):
    """
    This function constructs a layer of dilated RNN.
    Inputs:
        cell -- the dilation operations is implemented independent of the RNN cell.
            In theory, any valid tensorflow rnn cell should work.
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
        rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper. 
        input_length -- [b], int32 or int64
        scope -- variable scope.
    Outputs:
        outputs -- the outputs from the RNN.
    """
    b, d = tf.unstack(tf.shape(inputs[0]))
    n_steps = len(inputs)#t
    if rate < 0 or rate >= n_steps:#rate是一个int，代表本层的空洞数目
        raise ValueError('The \'rate\' variable needs to be adjusted.')
    print("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (
        scope, n_steps, rate, inputs[0].get_shape()[1]))

    # make the length of inputs divide 'rate', by using zero-padding
    EVEN = (n_steps % rate) == 0#检查是否可以整除
    if not EVEN:
        # Create a tensor in shape (batch_size, input_dims), which all elements are zero.  
        # This is used for zero padding
        zero_tensor = tf.zeros_like(inputs[0])#shape[b,d]
        dialated_n_steps = n_steps // rate + 1
        print("=====> %d time points need to be padded. " % (
            dialated_n_steps * rate - n_steps))
        print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))
        for i_pad in range(dialated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)#此时在imput列表中添加了几项，使得inputs的长度与rate能够整除
    else:
        dialated_n_steps = n_steps // rate
        print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))

    # now the length of 'inputs' divide rate
    # reshape it in the format of a list of tensors
    # the length of the list is 'dialated_n_steps' 
    # the shape of each tensor is [batch_size * rate, input_dims] 
    # by stacking tensors that "colored" the same

    # Example: 
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    
    
    #if n_steps is 5, rate is 2, inputs is [x0, x1, x2, x3, x4], zero padded inputs is [x0, x1, x2, x3, x4, x5=0], xi是[b,d]
    #dialated_n_steps is 3
    #dilated_inputs -[[x0,x1], [x2,x3], [x4,x5=0]]
    #                -> i=0 [x0,x1] [b*rate,d]以前是一个时刻输入一个样本，现在是一个时刻输入两个样本的concat，分别是同一条数据在rate个时间的值
    #                -> i=1 [x2,x3]
    #                -> i=2 [x4,x5]
    #
    #if rate = 3, dialated_n_steps is 2
    #dilated_inputs -[[x0,x1,x2],[x3,x4,x5=0]]
    #                -> i=0 [x0,x1,x2]
    #                -> i=1 [x3,x4,x5=0]
    #
    #
    #
    #
    # input shape, len = t, each of them is [b,d]
    # dilated_inputs shape, len = dialated_n_steps, each of them is [b*rate, d]
    #
    dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate], axis=0) for i in range(dialated_n_steps)]
            
    #dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate], axis=0) for i in range(dialated_n_steps)]
    #seq_len指示的是tf.concat(inputs[i * rate:(i + 1) * rate], axis=0) 序列中的有效位数

    # building a dialated RNN with reformated (dilated) inputs
    '''
    tf.nn.static_rnn(
        cell,
        inputs,
        initial_state=None,
        dtype=None,
        sequence_length=None,
        scope=None
    )
    Args:
            cell: An instance of RNNCell.
            inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
            initial_state: (optional) An initial state for the RNN. If cell.state_size is an integer, this must be a Tensor of appropriate type and shape [batch_size, cell.state_size]. If cell.state_size is a tuple, this should be a tuple of tensors having shapes [batch_size, s] for s in cell.state_size.
            dtype: (optional) The data type for the initial state and expected output. Required if initial_state is not provided or RNN state has a heterogeneous dtype.
            sequence_length: Specifies the length of each sequence in inputs. An int32 or int64 vector (tensor) size [batch_size], values in [0, T).
            scope: VariableScope for the created subgraph; defaults to "rnn".
    Returns:
            A pair (outputs, state) where:
            outputs is a length T list of outputs (one for each input), or a nested tuple of such elements.
            state is the final state
    '''
    #dilated_inputs shape, len = dialated_n_steps, each of them is [b*rate, d], 其实是将一个n_step的序列，变成了dialated_n_steps的, 即缩短了时间轴上的长度，
    #但是为了不丢失信息，每个时间步长上的数据增多为batch*rate
    #原来的结构中是一个时刻输入一个样本，一共输入n_step个时刻
    #现在是一个时刻输入rate个样本，一共输入n_step//rate个时刻
    
    #sequence_length: 由两个因素决定，一个是从placeholder中告知的，一个是前面的补0，其实应该由第一个因素来决定
    #dilated_states：应该是[b*rate,,hidden_units],意义应该是RNN层在并行的处理了这个batch的数据后的final state
    dilated_outputs, _ = tf.nn.static_rnn(#output shape ?? list, len = dialated_n_steps, each of them is [b*rate, hidden_units]
        cell, 
        dilated_inputs,
        dtype=tf.float32, 
        scope=scope)#dilated_states shape [b*rate, hidden_unit]

    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to 
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
    
    #list, len = dialated_n_steps, each of them is a list, len = rate, each of them is[b, hidden_units], 上面的作者注释应该写错了
    splitted_outputs = [tf.split(output, rate, axis=0)
                        for output in dilated_outputs]
    #list, len = dialated_n_steps * rate == n_step(就是补充过0之后的n_step数，应该是rate的倍数), each of them is [b, hidden_units]
    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]
    # remove padded zeros
    #list, len = n_step(不加pad的时候的数目), each of them is [b, hidden_units]
    outputs = unrolled_outputs[:n_steps]#可以增加返回状态的项，decoder要用
    
    dilated_states = outputs[-1]#对于GRU，state就是输出，所以只要截取最后一个时刻的输出即可

    return outputs, dilated_states


def multi_dRNN_with_dilations(cells, inputs, dilations, scope='drnn'):
    """
    This function constucts a multi-layer dilated RNN. 
    Inputs:
        cells -- A list of RNN cells.
        inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
        input_length -- [b]
        initial_states -- A list of final state in each encoder layer
    Outputs:
        x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
    """
    drnn_inputs = tf.transpose(inputs, [1,0,2])#[t,b,d]
    drnn_inputs = tf.unstack(drnn_inputs, axis=0)
    
    assert (len(cells) == len(dilations))

    output = copy.copy(drnn_inputs)
    final_state_list = []
    for cell, dilation in zip(cells, dilations):
        scope_name = "multi_dRNN_dilation_%s_%d" % (scope, dilation)
        output, state = dRNN(cell, output, dilation, scope=scope_name)
        final_state_list.append(state)
    output = tf.stack(output, axis=0)# [t,b,d]
    output = tf.transpose(output, [1,0,2]) #[b,t,d]
    return output, final_state_list

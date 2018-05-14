import tensorflow as tf 
import numpy as np

# n1 = np.random.rand(2,3)

n1=np.array([6.0] , dtype=np.float)

n2 = np.array([1.0], dtype=np.float)

input1_arr = np.array([1],dtype=np.int32)

t1 = tf.convert_to_tensor(n1)
t2 = tf.nn.softmax(t1)
t3 = tf.convert_to_tensor(n2)

t4 = tf.losses.softmax_cross_entropy(logits=t1, onehot_labels=t3)

print("debug")
with tf.Session() as sess:
    # feed={
    #     input1: input1_arr
    # }
    _t1,_t2, _t4= sess.run([t1,t2, t4])
    print(_t1)
    # print('input1',input1_arr)
    print('t2:',_t2)
    print('t4:', _t4)
    


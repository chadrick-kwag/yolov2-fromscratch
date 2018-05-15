import tensorflow as tf 
import numpy as np

# n1 = np.random.rand(2,3)

n1=np.random.rand(2,3,4)





t1 = tf.convert_to_tensor(n1)

t2 = tf.reduce_sum(t1, axis=0)

t3 = tf.reduce_sum(t1, axis=1)

t4 = tf.reduce_sum(t1, axis=2)

t5 = tf.reshape(t1,shape=[-1,12])

t6 = tf.reduce_sum(t5, axis=1)

t7 = tf.reduce_mean(t6)



print("debug")
with tf.Session() as sess:
    # feed={
    #     input1: input1_arr
    # }
    _t1, _t2, _t3, _t4, _t5, _t6, _t7= sess.run([t1,t2, t3, t4, t5, t6, t7])
    print('t1:',_t1)
    print('t2:',_t2)
    print('t3:',_t3)
    print('t4:',_t4)
    # print('input1',input1_arr)
    # print('t2:',_t2)
    # print('t4:', _t4)
    print('t5:', _t5)
    print('t6:', _t6)
    print('t7:', _t7)
    


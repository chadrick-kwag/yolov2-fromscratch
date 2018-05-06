import tensorflow as tf 
import numpy as np

# n1 = np.random.rand(2,3)

n1=np.array([ [1.0, 0.0] , [1.0 ,1.0]] , dtype=np.float)

l1 = np.array([ [0.5, 2], [0.1,3]  ])

input1_arr = np.array([1],dtype=np.int32)

t1 = tf.convert_to_tensor(n1)
t2 = tf.convert_to_tensor(l1)
input1 = tf.placeholder(tf.int32,shape=(1))

interm1= input1[0]

t3 = t2[interm1,:]
# t3=t2


t5 = 1- t1

# t3 = tf.losses.softmax_cross_entropy(t2,t1)

# t4 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t2,logits=t1)
print("debug")
with tf.Session() as sess:
    feed={
        input1: input1_arr
    }
    _t1,_t2,  _t5, _t3 = sess.run([t1,t2,t5,t3],feed_dict=feed)
    print(_t1)
    print('input1',input1_arr)
    print('t2:',_t2)
    # print(_t5)
    print("t3", _t3)


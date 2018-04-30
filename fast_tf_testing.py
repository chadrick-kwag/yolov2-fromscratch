import tensorflow as tf 
import numpy as np

# n1 = np.random.rand(2,3)

n1=np.array([ [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0] ])

l1 = np.array([0,0],dtype=int)

t1 = tf.convert_to_tensor(n1)
t2 = tf.convert_to_tensor(l1)

t5 = tf.nn.softmax(t1)

# t3 = tf.losses.softmax_cross_entropy(t2,t1)

t4 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t2,logits=t1)

with tf.Session() as sess:
    _t1,_t2, _t4, _t5 = sess.run([t1,t2,t4,t5])
    print(_t1)
    print(_t2)
    print(_t4)
    print(_t5)


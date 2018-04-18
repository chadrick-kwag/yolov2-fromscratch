"""
this code is for tracking and understanding the array flow of the
loss calculation in darkflow

"""

import numpy as np
import tensorflow as tf
# shape of [1,4,5,1]

HW=4
B=5

m1 = np.random.rand(1,HW,B,2)
m2 = np.random.rand(1,HW,B)

print("m1 shape={}".format(m1.shape))

intersect_wh = tf.convert_to_tensor(m1)
intersect = tf.multiply(intersect_wh[:,:,:,0],intersect_wh[:,:,:,1])

print("intersect_wh = {}".format(intersect_wh))
print("intersect = {}".format(intersect))

dividend = tf.convert_to_tensor(m2)

iou = tf.truediv(intersect,dividend)

reduced = tf.reduce_max(iou,[2],True)

print("iou ={}".format(iou))
print("reduced = {}".format(reduced))

bestbox2 = tf.equal(iou,reduced)
bestbox = tf.to_float(bestbox2)
bestbox = tf.cast(bestbox,tf.float64)

print("bestbox = {}".format(bestbox))

_confs = tf.convert_to_tensor(m2)

confs = tf.multiply(bestbox,_confs)

print("bestbox = {}".format(bestbox))
print("confs = {}".format(confs))

with tf.Session() as sess:
	a = sess.run([confs])
	print(a)
	
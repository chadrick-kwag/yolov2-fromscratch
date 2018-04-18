import numpy as np
import tensorflow as tf

# create network

# input_layer = tf.reshape(inputs,[-1,416,416,3])


input_layer = tf.placeholder(tf.float32,shape=(None,416,416,3))

conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(inputs=conv5,pool_size=[2,2],strides=2)

conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv7 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

conv8 = tf.layers.conv2d(inputs=conv7, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

pool4 = tf.layers.max_pooling2d(inputs=conv8,pool_size=[2,2],strides=2)

conv9 = tf.layers.conv2d(inputs=conv8, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv10 = tf.layers.conv2d(inputs=conv9, filters=256, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

conv11 = tf.layers.conv2d(inputs=conv10, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv12 = tf.layers.conv2d(inputs=conv11, filters=256, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

pool5 = tf.layers.max_pooling2d(inputs=conv13,pool_size=[2,2],strides=2)

conv14 = tf.layers.conv2d(inputs=pool5, filters=1024, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv15 = tf.layers.conv2d(inputs=conv14, filters=512, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

conv16 = tf.layers.conv2d(inputs=conv15, filters=1024, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv17 = tf.layers.conv2d(inputs=conv16, filters=512, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

conv18 = tf.layers.conv2d(inputs=conv17, filters=1024, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

### upto here is the darknet-19 except the last layer

## adding additional layers specific for detection

conv19 = tf.layers.conv2d(inputs=conv18, filters=1024, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv20 = tf.layers.conv2d(inputs=conv19, filters=1024, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

conv21 = tf.layers.conv2d(inputs=conv20, filters=1024, kernel_size=[3,3], padding="same", activation=tf.nn.relu)


# reason for 30: 5 boxes(for each anchor point) x 6(5(box coordinates+objectiveness)+1(class prediction. but one class))
conv22 = tf.layers.conv2d(inputs=conv21, filters=30, kernel_size=[1,1], padding="same", activation=tf.nn.relu)




with tf.Session() as sess:
	writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
	writer.flush()

print("end of code")


# graph_def = tf.get_default_graph().as_graph_def()


# init_op = tf.global_variables_initializer()

# saver = tf.train.Saver()	

# with tf.Session() as sess:
# 	sess.run(init_op)
# 	save_path = saver.save(sess.graph,"./model.ckpt")
# 	print("saved in {}".format(save_path))





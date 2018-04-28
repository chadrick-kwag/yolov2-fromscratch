from create_net import create_training_net
import tensorflow as tf


# load the input and GT

# input_batch
# gt_batch


g1 = create_training_net()

with tf.Session(graph=g1) as sess:
    writer= tf.summary.FileWriter(logdir="./summary",graph=sess.graph)
    writer.flush()

print("end of code")
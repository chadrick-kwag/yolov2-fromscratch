import numpy as np
import tensorflow as tf

# create network

# 

def create_training_net():

    graph = tf.Graph()

    with graph.as_default():

        input_layer = tf.placeholder(tf.float32,shape=(None,416,416,3),name="input_batch")
        # ground_truth = tf.placeholder(tf.float32, shape=(None,13,13,30),name="gt_batch")
        ground_truth = tf.placeholder(tf.float32, shape=(None,13*13,5,6),name="gt_batch")

        input_placeholders = {
            'input_layer': input_layer,
            'ground_truth': ground_truth
        }

        output = tf.placeholder(tf.float32,shape=(None,13,13,30))

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu, name="conv_chad_1")

        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu, name="conv_chad_2")

        pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2, name="pool_chad_2")

        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu, name="conv_chad_3")

        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[1,1], padding="same", activation=tf.nn.relu, name="conv_chad_4")

        conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(inputs=conv5,pool_size=[2,2],strides=2)

        conv6 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

        conv7 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

        conv8 = tf.layers.conv2d(inputs=conv7, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

        pool4 = tf.layers.max_pooling2d(inputs=conv8,pool_size=[2,2],strides=2)

        conv9 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

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


        # the output is [-1,13,13,30]
        net_out = tf.identity(conv22,name="net_out")

        net_out_reshaped = tf.reshape(net_out,[-1,13,13,5,6])

        raw_coords = tf.reshape(net_out_reshaped[:,:,:,:,:4],[-1,13*13,5,4])

        


        # apply logistic function to coordinate predictions

        # output of cx,cy should be 0< val < 1 since it is a relative value to the size of a single grid cell

        # bw,by should also be 0<val<1 since it is a relative value to the width and height of the image

        
        coords = tf.nn.sigmoid(raw_coords,name="coord_pred_op")
        coords = tf.identity(coords,name="coord_pred")

        raw_conf = tf.reshape(net_out_reshaped[:,:,:,:,4],[-1,13*13,5,1],name="raw_conf")
        conf = tf.nn.softmax(raw_conf,name="conf_pred_op")
        conf = tf.identity(conf,name="conf_pred")

        raw_pclass = tf.reshape(net_out_reshaped[:,:,:,:,5:],[-1,13*13,5,1],name="raw_pclass")
        pclass = tf.nn.softmax(raw_pclass,name="pclass_pred_op")
        pclass = tf.identity(pclass,name="pclass_pred")


        # get loss function
        # gt shape: [-1,13*13,5,6]
        
        gt_coords = tf.reshape(gt[:,:,:,0:4],[-1,13*13,5,4])
        gt_conf = tf.reshape(gt[:,:,:,4],[-1,13*13,5,1])
        gt_pclass = tf.reshape( gt[:,:,:,5], [-1,13*13,5,1] )

        


        notable_tensors={
            'coord_pred': coords,
            'conf_pred': conf,
            'pclass_pred' : pclass
        }



    return graph, notable_tensors, input_placeholders



        # """
        # create loss function
        # """

        # # all boxes have x,y,w,h,c,p values

        # # get the GT raw array

        # # get the loss
        # difference = GT - output

        # loss = difference ^2 # or something ...

        # # optimize it.








        # """
        # just trying to save the model here
        # """
        # with tf.Session() as sess:
        #     writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
        #     writer.flush()

        # print("end of code")


        # # graph_def = tf.get_default_graph().as_graph_def()


        # # init_op = tf.global_variables_initializer()

        # # saver = tf.train.Saver()	

        # # with tf.Session() as sess:
        # # 	sess.run(init_op)
        # # 	save_path = saver.save(sess.graph,"./model.ckpt")
        # # 	print("saved in {}".format(save_path))





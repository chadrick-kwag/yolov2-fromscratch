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

        essence = tf.placeholder(tf.float32,shape=(6),name="essence")

        

        gt_bbx_grid_index = tf.cast(essence[0],tf.int32)
        gt_bbx_box_index = tf.cast(essence[1],tf.int32)
        gt_bbx_coords = essence[2:]

        early_gt_poi_array = ground_truth[0,gt_bbx_grid_index,gt_bbx_box_index,:]


        ap_list= tf.placeholder(tf.float32,shape=(5,2),name="ap_list")

        input_placeholders = {
            'input_layer': input_layer,
            'ground_truth': ground_truth,
            'ap_list': ap_list,
            'essence': essence
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
        conv22 = tf.layers.conv2d(inputs=conv21, filters=30, kernel_size=[1,1], padding="same")


        # the output is [-1,13,13,30]
        net_out = tf.identity(conv22,name="net_out")

        net_out_reshaped = tf.reshape(net_out,[-1,13,13,5,6])

        raw_coords = tf.reshape(net_out_reshaped[:,:,:,:,:4],[-1,13*13,5,4])

        


        # apply logistic function to coordinate predictions

        # output of cx,cy should be 0< val < 1 since it is a relative value to the size of a single grid cell

        # bw,by should also be 0<val<1 since it is a relative value to the width and height of the image

        
        # coords = tf.nn.sigmoid(raw_coords,name="coord_pred_op")
        pred_raw_cxy = tf.reshape(raw_coords[:,:,:,0:2],[-1,13*13,5,2])
        pred_raw_wh = tf.reshape(raw_coords[:,:,:,2:4],[-1,13*13,5,2])

        pred_normalized_cxy = tf.nn.sigmoid(pred_raw_cxy)

        pred_before_ap_exped_wh = tf.exp(pred_raw_wh)
        pred_after_ap_normalized_wh = tf.multiply(pred_before_ap_exped_wh,ap_list)

        
        raw_conf = tf.reshape(net_out_reshaped[:,:,:,:,4],[-1,13*13,5,1],name="raw_conf")
        # conf = tf.nn.sigmoid(raw_conf,name="conf_pred_op")
        conf = raw_conf
        conf = tf.identity(conf,name="conf_pred")

        raw_pclass = tf.reshape(net_out_reshaped[:,:,:,:,5:],[-1,13*13,5,1],name="raw_pclass")
        pclass = tf.nn.softmax(raw_pclass,name="pclass_pred_op")
        pclass = tf.identity(pclass,name="pclass_pred")


        # get loss function
        # gt shape: [-1,13*13,5,6]
        
        gt_coords = tf.reshape(ground_truth[:,:,:,0:4],[-1,13*13,5,4])

        # the g_cxy and gt_wh are already given as values relative to grid_cell_size(w&h)
        gt_cxy = tf.reshape(gt_coords[:,:,:,0:2],[-1,13*13,5,2])
        gt_wh = tf.reshape(gt_coords[:,:,:,2:4],[-1,13*13,5,2])

        check1_poi_conf = ground_truth[0,gt_bbx_grid_index,gt_bbx_box_index,4]

        gt_conf = tf.reshape(ground_truth[:,:,:,4],[-1,13*13,5,1])

        check2_poi_conf = gt_conf[0,gt_bbx_grid_index, gt_bbx_box_index,0]

        
        gt_pclass = tf.reshape( ground_truth[:,:,:,5], [-1,13*13,5,1] )

        #============

        # need to get the mask of gt
        gt_mask = tf.equal(gt_conf,1.0)
        gt_mask = tf.to_float(gt_mask)
        gt_mask_true_count = tf.count_nonzero(gt_mask)


        #============

        # reminder: gt_cxy is already relative to grid_cell_size.
        # therefore it is okay to work with pred_normalized_cxy which is also
        # a value regarded to be relative to grid_cell_size

        # loss_cxy = tf.subtract(gt_cxy,pred_normalized_cxy)
        # loss_cxy = tf.pow(loss_cxy,2)
        # loss_cxy = tf.reduce_sum(loss_cxy,axis=1)
        # loss_cxy = tf.reduce_mean(loss_cxy)

        loss_cxy = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cxy, logits=pred_raw_cxy)
        loss_cxy = tf.reduce_sum(loss_cxy)
        #============

        # reminder: gt_wh is already relative to grid_cell_size
        # therefore it is okay to work with pred_after_ap_normalized_wh which is also
        # a value regarded to be relative to grid_cell_size
        loss_wh = tf.subtract(gt_wh, pred_after_ap_normalized_wh)
        loss_wh = tf.pow(loss_wh,2)
        loss_wh = tf.reduce_sum(loss_wh,axis=1)
        loss_wh = tf.reduce_mean(loss_wh)

        #============

        # loss_coords = loss_wh + loss_cxy
        loss_coords = loss_cxy

        #=============

        # loss_conf_1 = tf.subtract(gt_conf, conf)
        # loss_conf_2 = tf.pow(loss_conf_1,2)
        # loss_conf_3 = tf.reduce_sum(loss_conf_2,axis=1)
        # loss_conf = tf.reduce_mean(loss_conf_3)

        loss_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf,logits=conf)
        loss_conf = tf.reduce_sum(loss_conf)


        #==============

        loss_pclass_1 = tf.subtract(gt_pclass, pclass)
        loss_pclass_2 = tf.pow(loss_pclass_1,2)
        loss_pclass_3 = tf.reduce_sum(loss_pclass_2,axis=1)
        loss_pclass = tf.reduce_mean(loss_pclass_3)


        #===== total loss
        # loss = loss_coords + 100* loss_conf + 100 * loss_pclass
        loss = loss_coords
        


        #======= setup optimizer

        optimizer = tf.train.GradientDescentOptimizer(0.00001)

        optimizing_op = optimizer.minimize(loss)


        

        #============ accuracy calculation

        # calculate x1,x2,y1,y2
        
        # use gt_cxy, gt_wh

        # corner1: left top corner
        # corner2: right bottom corner

        gt_corner1 = gt_cxy - 0.5*gt_wh
        pred_corner1 = pred_normalized_cxy - 0.5 * pred_after_ap_normalized_wh

        gt_corner2 = gt_cxy + 0.5 * gt_wh
        pred_corner2 = pred_normalized_cxy + 0.5 * pred_after_ap_normalized_wh

        gt_area = gt_wh[:,:,:,0] * gt_wh[:,:,:,1]
        gt_area = tf.reshape(gt_area,shape=[-1,13*13,5,1])

        pred_area = pred_after_ap_normalized_wh[:,:,:,0] * pred_after_ap_normalized_wh[:,:,:,1]
        pred_area = tf.reshape(pred_area, shape = [-1,13*13,5,1])

        # copmare the corners and get the intersection

        # intersection_corner1: the maximum from the two corner1
        # intersection_corner2 : the minimum from the two corner2
        intersection_corner1 = tf.maximum(gt_corner1, pred_corner1)
        intersection_corner2 = tf.minimum(gt_corner2, pred_corner2)

        intersection_wh= tf.subtract(intersection_corner2,intersection_corner1)
        
        intersection_area = tf.multiply(intersection_wh[:,:,:,0], intersection_wh[:,:,:,1])
        intersection_area = tf.reshape(intersection_area,[-1,13*13,5,1])

        total_area = gt_area + pred_area - intersection_area

        # hmm... i'm a bit worried about the zero division...
        iou = tf.divide(intersection_area, total_area, name="iou_op")

        valid_iou_boolmask = tf.cast(tf.greater(iou,0.5),tf.float32,name="valid_iou_boolmask_op")
        valid_iou = tf.multiply(valid_iou_boolmask, iou, name="valid_iou_op")

        #====== get gt gt_box_exist_mask

        gt_box_exist_mask = tf.cast(tf.greater(gt_conf,0.5), tf.float32)
        gt_box_invert_exist_mask = 1.0 - gt_box_exist_mask

        # gt_box_count = tf.reduce_sum(gt_box_exist_mask)
        gt_box_count = tf.count_nonzero(gt_box_exist_mask)


        #====== filter valid iou with gt_box_exist_mask in order to filter out non-gt boxes

        # calculate correct_hit, incorrect_hit

        correct_hit_iou = tf.multiply(valid_iou, gt_box_exist_mask,name="correct_hit_iou_op")
        incorrect_hit_iou = tf.multiply(valid_iou, gt_box_invert_exist_mask,name="incorrect_hit_iou_op")

        # calculate the count for corrent and incorrect

        correct_hit_count = tf.count_nonzero(correct_hit_iou)
        incorrect_hit_count = tf.count_nonzero(incorrect_hit_iou)


        # get precision, recall 
        # for info: https://en.wikipedia.org/wiki/Precision_and_recall

        # precision: correct_hit / (correct_hit + incorrect_hit)
        # recall: corect_hit / gt_box_count

        precision = correct_hit_count / (correct_hit_count + incorrect_hit_count)
        recall = correct_hit_count / gt_box_count



        #===== debug_check

        debug_gtbbx_iou = iou[:,gt_bbx_grid_index,gt_bbx_box_index,0]
        debug_pred_normalized_cxy = pred_normalized_cxy[:,gt_bbx_grid_index, gt_bbx_box_index,:]
        debug_pred_after_ap_normalized_wh = pred_after_ap_normalized_wh[:,gt_bbx_grid_index, gt_bbx_box_index,:]

        debug_poi_cx= debug_pred_normalized_cxy[0,0]
        debug_poi_cy = debug_pred_normalized_cxy[0,1]
        debug_poi_rw= debug_pred_after_ap_normalized_wh[0,0]
        debug_poi_rh = debug_pred_after_ap_normalized_wh[0,1]
        debug_poi_iou = debug_gtbbx_iou[0]

        debug_gt_poi_conf = gt_conf[0,gt_bbx_grid_index,gt_bbx_box_index,0]



        #========= setup summary

        tf.summary.scalar(name="loss_coords",tensor=loss_coords)
        tf.summary.scalar(name="loss_conf",tensor=loss_conf)
        tf.summary.scalar(name="loss_pclass",tensor=loss_pclass)
        tf.summary.scalar(name="loss",tensor=loss)
        tf.summary.scalar(name="debug_poi_cx",tensor=debug_poi_cx)
        tf.summary.scalar(name="debug_poi_cy", tensor=debug_poi_cy)
        tf.summary.scalar(name="debug_poi_rw", tensor=debug_poi_rw)
        tf.summary.scalar(name="debug_poi_rh", tensor=debug_poi_rh)
        tf.summary.scalar(name="debug_poi_iou",tensor =debug_poi_iou )

        summary_op = tf.summary.merge_all()
    
        notable_tensors={
            'conf_pred': conf,
            'pclass_pred' : pclass,
            'loss_coords': loss_coords,
            'total_loss': loss,
            'optimizing_op': optimizing_op,
            'summary_op' : summary_op,
            'gt_box_count' : gt_box_count,
            'correct_hit_count': correct_hit_count,
            'incorrect_hit_count' : incorrect_hit_count,
            'precision' : precision,
            'recall': recall,
            'iou': iou,
            'valid_iou_boolmask': valid_iou_boolmask,
            'gt_bbx_grid_index': gt_bbx_grid_index,
            'gt_bbx_box_index': gt_bbx_box_index,
            'gt_bbx_coords': gt_bbx_coords,
            'debug_gtbbx_iou': debug_gtbbx_iou,
            'debug_pred_normalized_cxy': debug_pred_normalized_cxy,
            'debug_pred_after_ap_normalized_wh': debug_pred_after_ap_normalized_wh,
            'gt_mask_true_count': gt_mask_true_count,
            'debug_gt_poi_conf':debug_gt_poi_conf,
            'early_gt_poi_array':early_gt_poi_array,
            'check1_poi_conf': check1_poi_conf,
            'check2_poi_conf': check2_poi_conf
        }



    return graph, notable_tensors, input_placeholders



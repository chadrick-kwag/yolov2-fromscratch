import numpy as np
import tensorflow as tf

# create network

# 

def create_training_net(istraining=True, debug_train_single_input = False):

    

    graph = tf.Graph()

    with graph.as_default():

        input_layer = tf.placeholder(tf.float32,shape=(None,416,416,3),name="input_batch")

        
        ground_truth = tf.placeholder(tf.float32, shape=(None,13*13,5,6),name="gt_batch")

        essence = tf.placeholder(tf.float32,shape=(6),name="essence")

        loss_weights_ph = tf.placeholder(tf.float32, shape=(5), name="loss_weights")
        loss_weights = tf.Print(loss_weights_ph, [loss_weights_ph], "loss_weights_ph=")
        
        lr_ph = tf.placeholder(tf.float32, shape=(1), name="lr_ph")        
        lr = tf.Print(lr_ph, [lr_ph], "loaded lr=")
        learning_rate = lr[0]

        # loss_weights = tf.Print(loss_weights, [loss_weights], "loaded loss_weights=")
        # lr = tf.Print(lr, [lr], "loaded lr=")
        learning_rate = tf.Print(learning_rate, [learning_rate], "learning rate=")
        
        
        

        gt_bbx_grid_index = tf.cast(essence[0],tf.int32)
        gt_bbx_box_index = tf.cast(essence[1],tf.int32)
        gt_bbx_coords = essence[2:]

        early_gt_poi_array = ground_truth[0,gt_bbx_grid_index,gt_bbx_box_index,:]


        ap_list= tf.placeholder(tf.float32,shape=(5,2),name="ap_list")

        input_placeholders = {
            'input_layer': input_layer,
            'ground_truth': ground_truth,
            'ap_list': ap_list,
            'essence': essence,
            'loss_weights': loss_weights_ph,
            'learning_rate' : lr_ph
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

        debug_pred_raw_poi_w = pred_raw_wh[0,gt_bbx_grid_index, gt_bbx_box_index, 0]
        debug_pred_raw_poi_h = pred_raw_wh[0,gt_bbx_grid_index, gt_bbx_box_index, 1]


        pred_normalized_cxy = tf.nn.sigmoid(pred_raw_cxy)

        

        pred_before_ap_exped_wh = tf.exp(pred_raw_wh)

        debug_pred_after_exp_poi_w = pred_before_ap_exped_wh[0, gt_bbx_grid_index, gt_bbx_box_index, 0]
        debug_pred_after_exp_poi_h = pred_before_ap_exped_wh[0, gt_bbx_grid_index, gt_bbx_box_index, 1]

        pred_after_ap_normalized_wh = tf.multiply(pred_before_ap_exped_wh,ap_list)

        check_poi_pred_w = pred_after_ap_normalized_wh[0,gt_bbx_grid_index, gt_bbx_box_index, 0 ]
        check_poi_pred_h = pred_after_ap_normalized_wh[0,gt_bbx_grid_index, gt_bbx_box_index, 1 ]

        
        raw_conf = tf.reshape(net_out_reshaped[:,:,:,:,4],[-1,13*13,5,1],name="raw_conf")
        # conf = tf.nn.sigmoid(raw_conf,name="conf_pred_op")
        conf = raw_conf
        conf = tf.identity(conf,name="conf_pred")
        pred_conf = tf.nn.sigmoid(conf)




        raw_pclass = tf.reshape(net_out_reshaped[:,:,:,:,5:],[-1,13*13,5,1],name="raw_pclass")
        pclass = tf.nn.softmax(raw_pclass,name="pclass_pred_op")
        pclass = tf.identity(pclass,name="pclass_pred")

        pclass_poi = pclass[:,gt_bbx_grid_index, gt_bbx_box_index, :]
        pclass_poi = tf.Print(pclass_poi, [pclass_poi], "pclass_poi:")


        # get loss function
        # gt shape: [-1,13*13,5,6]
        
        gt_coords = tf.reshape(ground_truth[:,:,:,0:4],[-1,13*13,5,4])

        # the g_cxy and gt_wh are already given as values relative to grid_cell_size(w&h)
        gt_cxy = tf.reshape(gt_coords[:,:,:,0:2],[-1,13*13,5,2])
        gt_cx = tf.reshape(gt_coords[:,:,:,0], [-1,13*13,5,1])
        gt_cy = tf.reshape(gt_coords[:,:,:,1], [-1,13*13,5,1])


        gt_wh = tf.reshape(gt_coords[:,:,:,2:4],[-1,13*13,5,2])
        gt_rw = tf.reshape(gt_coords[:,:,:,2],[-1,13*13,5,1])
        gt_rh = tf.reshape(gt_coords[:,:,:,3],[-1,13*13,5,1])

        check1_poi_conf = ground_truth[0,gt_bbx_grid_index,gt_bbx_box_index,4]
        
        check_poi_gt_w = gt_wh[0,gt_bbx_grid_index, gt_bbx_box_index, 0]
        check_poi_gt_h = gt_wh[0,gt_bbx_grid_index, gt_bbx_box_index, 1]


        gt_conf = tf.reshape(ground_truth[:,:,:,4],[-1,13*13,5,1])

        check2_poi_conf = gt_conf[0,gt_bbx_grid_index, gt_bbx_box_index,0]

        
        gt_pclass = tf.reshape( ground_truth[:,:,:,5], [-1,13*13,5,1] )

        #============

        # need to get the mask of gt
        gt_mask_boolean = tf.equal(gt_conf,1.0)
        gt_mask = tf.to_float(gt_mask_boolean)
        gt_mask_invert_float = 1.0 - gt_mask
        gt_mask_true_count = tf.count_nonzero(gt_mask)


        #============

        # reminder: gt_cxy is already relative to grid_cell_size.
        # therefore it is okay to work with pred_normalized_cxy which is also
        # a value regarded to be relative to grid_cell_size


        #lets mask it
        pred_raw_cxy_masked = pred_raw_cxy * gt_mask

        
        pred_raw_cx_masked = tf.reshape(pred_raw_cxy_masked[:,:,:,0], shape=[-1,13*13,5,1])
        pred_raw_cy_masked = tf.reshape(pred_raw_cxy_masked[:,:,:,1], shape=[-1,13*13,5,1])

        loss_cx = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cx, logits = pred_raw_cx_masked)
        loss_cx = loss_cx * gt_mask
        loss_cx = tf.reduce_sum(loss_cx, axis=1)
        loss_cx = tf.reduce_mean(loss_cx)

        loss_cy = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cy, logits = pred_raw_cy_masked)
        loss_cy = loss_cy * gt_mask
        loss_cy = tf.reduce_sum(loss_cy, axis=1)
        loss_cy = tf.reduce_mean(loss_cy)



        

        loss_cxy_array = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cxy, logits=pred_raw_cxy_masked)
        loss_cxy_array = loss_cxy_array * gt_mask


        loss_cxy = tf.reshape(loss_cxy_array, shape=[-1,13*13*5*2])
        loss_cxy = tf.reduce_sum(loss_cxy,axis=1)
        loss_cxy = tf.reduce_mean(loss_cxy)
        
        loss_cxy = tf.Print(loss_cxy, [loss_cxy], "loss_cxy=")

        loss_cxy_poi = loss_cxy_array[:,gt_bbx_grid_index, gt_bbx_box_index, :]
        # loss_cxy_poi = tf.Print(loss_cxy_poi,[loss_cxy_poi], "loss_cxy_poi")
        #============



        pred_wh_masked = pred_after_ap_normalized_wh * gt_mask

        pred_rw  = tf.reshape(pred_wh_masked[:,:,:,0], shape=[-1,13*13,5,1])
        loss_rw = tf.losses.mean_squared_error(labels=gt_rw, predictions= pred_rw, reduction=tf.losses.Reduction.MEAN)
        

        pred_rh = tf.reshape(pred_wh_masked[:,:,:,1], shape=[-1,13*13,5,1])
        loss_rh = tf.losses.mean_squared_error(labels=gt_rh, predictions= pred_rh, reduction=tf.losses.Reduction.MEAN)
        

        loss_wh = tf.losses.mean_squared_error(labels=gt_wh, predictions= pred_wh_masked)
        loss_wh = tf.Print(loss_wh, [loss_wh], "loss_wh=")

  

        #============

        loss_coords = loss_cxy + loss_wh
        # loss_coords = loss_wh

        #=============

        pred_conf_poi = conf[:,gt_bbx_grid_index, gt_bbx_box_index, :]
        # pred_conf_poi = tf.Print(pred_conf_poi, [pred_conf_poi], "pred_conf_poi:")

        gt_conf_poi = gt_conf[:,gt_bbx_grid_index, gt_bbx_box_index, :]
        # gt_conf_poi = tf.Print(gt_conf_poi, [gt_conf_poi], "gt_conf_poi:")

        

        # conf_masked = conf * gt_mask

        # conf_threshold = 0.3
        # over_threshold_conf_mask  = conf > 0.3
        # over_threshold_conf_mask = over_threshold_conf_mask 
        # over_threshold_conf_mask_float = tf.to_float(over_threshold_conf_mask)
        # # remove the gt from over_threshold
        # over_threshold_conf_mask_float -= gt_mask

        # threshold_converted = 0.3* over_threshold_conf_mask_float

        # gt_masked_conf = conf * gt_mask

        # target_conf = threshold_converted + gt_masked_conf

        loss_conf_weight = 0.1*gt_mask_invert_float + 10 * gt_mask
        print("loss_conf_weight", loss_conf_weight)

        loss_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf,logits=conf)
        loss_conf = tf.multiply(loss_conf, loss_conf_weight)
        print("loss_conf after multiplying with loss_conf_weight:", loss_conf)
        loss_conf = tf.reshape(loss_conf, shape=[-1,13*13*5])
        loss_conf = tf.reduce_sum(loss_conf, axis=1)
        loss_conf = tf.reduce_mean(loss_conf)
        loss_conf = tf.Print(loss_conf, [loss_conf], "loss_conf=")


        #==============pclass legacy code
        

        # loss_pclass = tf.nn.softmax_cross_entropy_with_logits(labels=gt_pclass,logits=raw_pclass)
        
        # loss_pclass_poi = loss_pclass[:, gt_bbx_grid_index, gt_bbx_box_index]
        # loss_pclass_poi = tf.Print(loss_pclass_poi, [loss_pclass_poi], "loss_pclass_poi=")
        
        # gt_mask_reshaped = tf.reshape(gt_mask,shape=[-1,13*13,5])

        # loss_pclass = loss_pclass * gt_mask_reshaped

        # loss_pclass = tf.reduce_sum(loss_pclass)




        #===== total loss
     
        #print losses
        loss_cx = tf.Print(loss_cx, [loss_cx], "loss_cx=")
        loss_cy = tf.Print(loss_cy, [loss_cy], "loss_cy=")
        loss_rw = tf.Print(loss_rw, [loss_rw], "loss_rw=")
        loss_rh = tf.Print(loss_rh, [loss_rh], "loss_rh=")
        loss_conf = tf.Print(loss_conf, [loss_conf], "loss_conf=")

        # loss = loss_coords + loss_conf
        loss = loss_weights[0]* loss_cx + loss_weights[1]*loss_cy + loss_weights[2] *loss_rw + \
            loss_weights[3]* loss_rh + loss_weights[4]*loss_conf
        loss = tf.Print(loss,[loss], "loss=")




        #======= setup optimizer
        # learning rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        optimizing_op = optimizer.minimize(loss)


        

        #============ accuracy calculation


        # filter valid_conf
        conf_threshold = 0.8
        valid_conf_mask = pred_conf > conf_threshold
        valid_conf_mask = tf.cast(valid_conf_mask, tf.float32)

        valid_conf_mask_count = tf.count_nonzero(valid_conf_mask)





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
        intersection_area = intersection_area + 1e-14

        total_area = gt_area + pred_area - intersection_area

        # hmm... i'm a bit worried about the zero division...
        iou = tf.divide(intersection_area, total_area, name="iou_op")

        # mask it with valid_conf_mask
        valid_conf_mask_applied_iou = valid_conf_mask * iou

        # get valid_iou mask
        iou_threshold = 0.5
        iou_mask = valid_conf_mask_applied_iou > iou_threshold
        iou_mask = tf.cast(iou_mask, tf.float32)

        filtered_iou = valid_conf_mask_applied_iou * iou_mask


        # valid_iou_boolmask = tf.cast(tf.greater(iou,0.5),tf.float32,name="valid_iou_boolmask_op")
        # valid_iou = tf.multiply(valid_iou_boolmask, iou, name="valid_iou_op")
        valid_iou = filtered_iou

        #====== get gt gt_box_exist_mask

        gt_box_exist_mask = tf.cast(tf.greater(gt_conf,0.5), tf.float32)
        gt_box_invert_exist_mask = 1.0 - gt_box_exist_mask

        # gt_box_count = tf.reduce_sum(gt_box_exist_mask)
        gt_box_count = tf.count_nonzero(gt_box_exist_mask)


        #====== filter valid iou with gt_box_exist_mask in order to filter out non-gt boxes

        # calculate correct_hit, incorrect_hit

        # correct_hit_iou = tf.multiply(valid_iou, gt_box_exist_mask,name="correct_hit_iou_op")
        
        # incorrect_hit_iou = tf.multiply(valid_iou, gt_box_invert_exist_mask,name="incorrect_hit_iou_op")


        
        pred_count = tf.count_nonzero(valid_iou)
        correct_hit = valid_iou * gt_mask
        correct_hit_count = tf.count_nonzero(correct_hit)
        incorrect_hit_count = pred_count - correct_hit_count

        precision = correct_hit_count / pred_count
        recall = correct_hit_count / gt_box_count

        



        poi_iou = iou * gt_mask
        poi_iou_rawform = poi_iou
        # sine there will be only one object, we simplify the caculation
        poi_iou = tf.reshape(poi_iou, shape=[-1,13*13*5])
        poi_iou = tf.reduce_sum(poi_iou,axis=1)
        poi_iou = tf.Print(poi_iou,[poi_iou],"poi_iou_vector")
        poi_iou_average = tf.reduce_mean(poi_iou)
        


        # correct_hit_iou_average = tf.reduce_sum(correct_hit_iou) / tf.cast(correct_hit_count, tf.float32)



        # get precision, recall 
        # for info: https://en.wikipedia.org/wiki/Precision_and_recall

        # precision: correct_hit / (correct_hit + incorrect_hit)
        # recall: corect_hit / gt_box_count

        #==== summarize conf

        
        poi_pred_conf = pred_conf * gt_mask
        # assuming that gt_mask will only leave one conf value alive...
        poi_pred_conf_average = tf.reshape(poi_pred_conf, shape=[-1,13*13*5])
        poi_pred_conf_average = tf.reduce_sum(poi_pred_conf_average, axis=1)
        poi_pred_conf_average = tf.reduce_mean(poi_pred_conf_average)
        poi_pred_conf_average = tf.Print(poi_pred_conf_average,[poi_pred_conf_average], "poi_pred_conf_average=")


        #===== debug_check

        debug_gtbbx_iou = iou[:,gt_bbx_grid_index,gt_bbx_box_index,0]
        debug_pred_normalized_cxy = pred_normalized_cxy[:,gt_bbx_grid_index, gt_bbx_box_index,:]
        debug_pred_after_ap_normalized_wh = pred_after_ap_normalized_wh[:,gt_bbx_grid_index, gt_bbx_box_index,:]

        debug_poi_cx= debug_pred_normalized_cxy[0,0]
        debug_poi_cy = debug_pred_normalized_cxy[0,1]
        debug_poi_rw= debug_pred_after_ap_normalized_wh[0,0]
        debug_poi_rh = debug_pred_after_ap_normalized_wh[0,1]
        debug_poi_iou = debug_gtbbx_iou[0]

        debug_poi_iou = tf.Print(debug_poi_iou, [debug_poi_iou], "debug_poi_iou=")

        debug_gt_poi_conf = gt_conf[0,gt_bbx_grid_index,gt_bbx_box_index,0]



        #========= setup summary
        # tf.summary.scalar(name="loss_cxy", tensor= loss_cxy)
        # tf.summary.scalar(name="loss_rwh", tensor = loss_wh)
        # tf.summary.scalar(name="loss_coords",tensor=loss_coords)
        # tf.summary.scalar(name="loss_pclass",tensor=loss_pclass)
        
        tf.summary.scalar(name="loss_cx",tensor=loss_cx)
        tf.summary.scalar(name="loss_cy", tensor=loss_cy)
        tf.summary.scalar(name="loss_rw", tensor = loss_rw)
        tf.summary.scalar(name="loss_rh", tensor= loss_rh)
        tf.summary.scalar(name="loss_conf",tensor=loss_conf)
        tf.summary.scalar(name="loss",tensor=loss)

        if debug_train_single_input:
            tf.summary.scalar(name="debug_poi_cx",tensor=debug_poi_cx)
            tf.summary.scalar(name="debug_poi_cy", tensor=debug_poi_cy)
            tf.summary.scalar(name="debug_poi_rw", tensor=debug_poi_rw)
            tf.summary.scalar(name="debug_poi_rh", tensor=debug_poi_rh)
            tf.summary.scalar(name="debug_poi_iou",tensor =debug_poi_iou )

        tf.summary.scalar(name="precision", tensor = precision)
        tf.summary.scalar(name="recall", tensor = recall)
        tf.summary.scalar(name="poi_iou_average", tensor=poi_iou_average)
        tf.summary.scalar(name="incorrect_hit_count", tensor=incorrect_hit_count)
        tf.summary.scalar(name="correct_hit_count", tensor = correct_hit_count)
        tf.summary.scalar(name="poi_pred_conf_average", tensor=poi_pred_conf_average)
        # tf.summary.scalar(name="correct_hit_iou_average", tensor = correct_hit_iou_average)



        summary_op = tf.summary.merge_all()
    
        notable_tensors={
            'conf_pred': conf,
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
            'check2_poi_conf': check2_poi_conf,
            'check_poi_pred_w': check_poi_pred_w,
            'check_poi_pred_h': check_poi_pred_h,
            'check_poi_gt_w': check_poi_gt_w,
            'check_poi_gt_h': check_poi_gt_h,
            'debug_pred_raw_poi_w': debug_pred_raw_poi_w,
            'debug_pred_raw_poi_h': debug_pred_raw_poi_h,
            'debug_pred_after_exp_poi_w': debug_pred_after_exp_poi_w,
            'debug_pred_after_exp_poi_h': debug_pred_after_exp_poi_h,
            'loss_cxy_poi': loss_cxy_poi,
            'pred_conf_poi': pred_conf_poi,
            'gt_conf_poi': gt_conf_poi,
            'gt_mask': gt_mask,
            'poi_iou': poi_iou,
            'poi_iou_rawform': poi_iou_rawform,
            'pred_out_cxy': pred_normalized_cxy,
            'pred_out_rwh' : pred_after_ap_normalized_wh,
            'pred_out_conf' : pred_conf,
            'poi_pred_conf_average': poi_pred_conf_average,
            'loss_cx': loss_cx,
            'loss_cy': loss_cy,
            'loss_rw' : loss_rw,
            'loss_rh' : loss_rh,
            'loss_conf' : loss_conf
            
        }



    return graph, notable_tensors, input_placeholders



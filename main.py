from create_net import create_training_net
import tensorflow as tf
import numpy as np
from inputloader import InputLoader
import os, pprint, sys

# load the input and GT


np_array_save_file = 'NP_SAVE.txt'

# input_batch
# gt_batch
STEP_NUM = 1000000
# STEP_NUM = 1

istraining = False


if istraining:
    g1,notable_tensors, input_holders = create_training_net()
else:
    g1,notable_tensors, input_holders = create_training_net(istraining=False)

# inputloader = InputLoader(testcase=0)

if istraining:

    inputloader = InputLoader(batch_num=5)

    image_input, gt, _ , essence = inputloader.get_image_and_gt()
else:
    inputloader = InputLoader(testcase=1)
    image_input, gt , _ , essence = inputloader.get_image_and_gt()


# essence format: ((center_xy_grid_index,best_B_index,r_cx,r_cy,resized_bw,resized_bh))

# print("essence={}".format(essence))


# check gt
# selected_essence = essence[0][0]
# gt_poi_grid_index = selected_essence[0]
# gt_poi_box_index = selected_essence[1]

# print('gt_poi_grid_index',gt_poi_grid_index)
# print('gt_poi_box_index',gt_poi_box_index)

# gt_poi_conf = gt[0][gt_poi_grid_index][gt_poi_box_index][4]
# print('gt_poi_conf',gt_poi_conf)




ap_list = inputloader.get_ap_list()

# print('ap_list',ap_list)
attempt_num=1
attempt_num_padded="{:03d}".format(attempt_num)
SAVE_PATH="./ckpt/model-{}.ckpt".format(attempt_num_padded)


#=== setup gpu configuration
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=False)
config = tf.ConfigProto(gpu_options=gpu_options)

# config.gpu_options.per_process_gpu_memory_fraction = 0.5



with tf.Session(graph=g1,config=config) as sess:


    if istraining:
        writer= tf.summary.FileWriter(logdir="./summary",graph=sess.graph)
        writer.flush()



    

    # wanted_outputs={
    #     'net_out': g1.get_tensor_by_name("net_out"),
    #     'coord_pred' : g1.get_operation_by_name("coord_pred")
    # }

    steps = STEP_NUM

    print('essence:',essence)
    print('essence to pass on to:', essence[0][0])

    feed_dict = {
        input_holders['input_layer'] : image_input,
        input_holders['ground_truth'] : gt,
        input_holders['ap_list'] : ap_list,
        input_holders['essence']: essence[0][0]
    }

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

        # if ckpt exist, then load from it
    
        # saver.restore(sess, SAVE_PATH)
    # last_checkpoint = tf.train.latest_checkpoint('./ckpt/')
    # print("last_checkpoint=", last_checkpoint)
    last_checkpoint = "./ckpt/model-001.ckpt-3000"
    if last_checkpoint is not None:
        saver.restore(sess,last_checkpoint)
        print("!!!!! restoring...!!! ")


    if istraining:
        fetches=[ notable_tensors['conf_pred'], 
            notable_tensors['loss_coords'],
            notable_tensors['optimizing_op'],
            notable_tensors['summary_op'],
            notable_tensors['precision'],
            notable_tensors['recall'],
            notable_tensors['gt_box_count'],
            notable_tensors['correct_hit_count'],
            notable_tensors['incorrect_hit_count'],
            notable_tensors['total_loss'],
            notable_tensors['iou'],
            notable_tensors['valid_iou_boolmask'],
            notable_tensors['gt_bbx_grid_index'],
            notable_tensors['gt_bbx_box_index'],
            notable_tensors['gt_bbx_coords'],
            notable_tensors['debug_gtbbx_iou'],
            notable_tensors['debug_pred_normalized_cxy'],
            notable_tensors['debug_pred_after_ap_normalized_wh'],
            notable_tensors['gt_mask_true_count'],
            notable_tensors['debug_gt_poi_conf'],
            notable_tensors['early_gt_poi_array'],
            notable_tensors['check1_poi_conf'],
            notable_tensors['check2_poi_conf'],
            notable_tensors['check_poi_pred_w'],
            notable_tensors['check_poi_pred_h'],
            notable_tensors['check_poi_gt_w'],
            notable_tensors['check_poi_gt_h'],
            notable_tensors['total_loss'],
            notable_tensors['debug_pred_raw_poi_w'],
            notable_tensors['debug_pred_raw_poi_h'],
            notable_tensors['debug_pred_after_exp_poi_w'],
            notable_tensors['debug_pred_after_exp_poi_h'],
            notable_tensors['loss_cxy_poi'],
            notable_tensors['pred_conf_poi'],
            notable_tensors['gt_conf_poi'],
            notable_tensors['gt_mask'],
            notable_tensors['poi_iou'],
            notable_tensors['poi_iou_rawform']
        
            ]
    else:
        fetches=[
            notable_tensors['pred_out_cxy'],
            notable_tensors['pred_out_rwh'],
            notable_tensors['pred_out_conf']
        ]


    try:

        if istraining:

            for step in range(steps):

                
                pred_conf, loss_coords, _ , summary_result, \
                    precision, recall, gt_box_count, correct_hit_count, incorrect_hit_count, \
                    loss, iou, valid_iou_boolmask , gt_bbx_grid_index,gt_bbx_box_index, gt_bbx_coords \
                    , debug_gtbbx_iou , debug_pred_normalized_cxy,debug_pred_after_ap_normalized_wh \
                    ,gt_mask_true_count, debug_gt_poi_conf, early_gt_poi_array, \
                    check1_poi_conf, check2_poi_conf, \
                    check_poi_pred_w, check_poi_pred_h ,\
                    check_poi_gt_w, check_poi_gt_h, total_loss, \
                    debug_pred_raw_poi_w, debug_pred_raw_poi_h ,\
                    debug_pred_after_exp_poi_w, debug_pred_after_exp_poi_h \
                    , loss_cxy_poi , pred_conf_poi, gt_conf_poi, gt_mask, poi_iou\
                    , poi_iou_rawform \
                    = sess.run(fetches,feed_dict=feed_dict)

                
                writer.add_summary(summary_result,global_step=step)
                


                
                pprint.pprint('step={} loss={}, precision={}, recall={}, gt_box_count={}, correct_hit_count={}, incorrect_hit_count={}'.format(
                    step,loss,precision,recall, gt_box_count, 
                    correct_hit_count, incorrect_hit_count))
                

                # np.savez(np_array_save_file,gt_mask = gt_mask, iou = iou, poi_iou = poi_iou, poi_iou_rawform = poi_iou_rawform)
                # print("gt_mask saved")

                # after all the steps, save to ckpt
                if step % 1000 ==0:
                    save_path = saver.save(sess,SAVE_PATH, global_step=step)
                    print("model saved to {}".format(save_path))

            print("train looping finished")
        
        else:
            # inferencing
            pred_out_cxy, pred_out_rwh, pred_out_conf = sess.run(fetches, feed_dict=feed_dict)

            print("pred_out_cxy shape=", pred_out_cxy.shape)
            print("pred_out_rwh shape=", pred_out_rwh.shape)
            print("pred_out_conf shape=", pred_out_conf.shape)

            np.savez("pred_save",pred_out_cxy = pred_out_cxy, pred_out_rwh = pred_out_rwh, pred_out_conf = pred_out_conf)
            print("prediction arrays saved as file")

            

    except:
        print("exception occured. exiting")




print("end of code")
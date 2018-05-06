from create_net import create_training_net
import tensorflow as tf
from inputloader import InputLoader
import os, pprint, sys

# load the input and GT

# input_batch
# gt_batch


g1,notable_tensors, input_holders = create_training_net()

inputloader = InputLoader(testcase=0)

image_input, gt, _ , essence = inputloader.get_image_and_gt()

# essence format: ((center_xy_grid_index,best_B_index,r_cx,r_cy,resized_bw,resized_bh))

print("essence={}".format(essence))


# check gt
selected_essence = essence[0][0]
gt_poi_grid_index = selected_essence[0]
gt_poi_box_index = selected_essence[1]

print('gt_poi_grid_index',gt_poi_grid_index)
print('gt_poi_box_index',gt_poi_box_index)

gt_poi_conf = gt[0][gt_poi_grid_index][gt_poi_box_index][4]
print('gt_poi_conf',gt_poi_conf)




ap_list = inputloader.get_ap_list()

# print('ap_list',ap_list)
SAVE_PATH="./ckpt/model.ckpt"


#=== setup gpu configuration
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=False)
config = tf.ConfigProto(gpu_options=gpu_options)

# config.gpu_options.per_process_gpu_memory_fraction = 0.5



with tf.Session(graph=g1,config=config) as sess:
    writer= tf.summary.FileWriter(logdir="./summary",graph=sess.graph)
    writer.flush()



    

    # wanted_outputs={
    #     'net_out': g1.get_tensor_by_name("net_out"),
    #     'coord_pred' : g1.get_operation_by_name("coord_pred")
    # }

    steps = 1000

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
    if os.path.exists(SAVE_PATH):
        saver.restore(sess, SAVE_PATH)


    fetches=[ notable_tensors['conf_pred'], 
        notable_tensors['pclass_pred'],
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
        notable_tensors['check2_poi_conf']

        ]

    for step in range(steps):
        pred_conf, pclass_pred, loss_coords, _ , summary_result, \
            precision, recall, gt_box_count, correct_hit_count, incorrect_hit_count, \
            loss, iou, valid_iou_boolmask , gt_bbx_grid_index,gt_bbx_box_index, gt_bbx_coords \
            , debug_gtbbx_iou , debug_pred_normalized_cxy,debug_pred_after_ap_normalized_wh \
            ,gt_mask_true_count, debug_gt_poi_conf, early_gt_poi_array, \
            check1_poi_conf, check2_poi_conf \
            = sess.run(fetches,feed_dict=feed_dict)

        writer.add_summary(summary_result,global_step=step)
        # print('coord_preds', coord_pred)
        pprint.pprint('step={} loss_conf={}, precision={}, recall={}, gt_box_count={}, correct_hit_count={}, incorrect_hit_count={}'.format(
            step,loss,precision,recall, gt_box_count, 
            correct_hit_count, incorrect_hit_count))

        print('gt_bbx_grid_index',gt_bbx_grid_index)
        print('gt_bbx_box_index',gt_bbx_box_index)
        print('gt_bbx_coords',gt_bbx_coords)
        print('debug_gtbbx_iou',debug_gtbbx_iou)
        print('debug_pred_normalized_cxy', debug_pred_normalized_cxy)
        print('debug_pred_after_ap_normalized_wh', debug_pred_after_ap_normalized_wh)
        print('gt_mask_true_count',gt_mask_true_count)
        print('debug_gt_poi_conf',debug_gt_poi_conf)
        print('early_gt_poi_array',early_gt_poi_array)
        print('check1_poi_conf',check1_poi_conf)
        print('check2_poi_conf',check2_poi_conf)
    

    # after all the steps, save to ckpt
    save_path = saver.save(sess,SAVE_PATH)
    print("model saved to {}".format(save_path))



print("end of code")
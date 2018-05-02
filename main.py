from create_net import create_training_net
import tensorflow as tf
from inputloader import InputLoader
import os, pprint

# load the input and GT

# input_batch
# gt_batch


g1,notable_tensors, input_holders = create_training_net()

inputloader = InputLoader()

image_input, gt, _ = inputloader.get_image_and_gt()

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

    steps = 100

    feed_dict = {
        input_holders['input_layer'] : image_input,
        input_holders['ground_truth'] : gt,
        input_holders['ap_list'] : ap_list
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
        notable_tensors['valid_iou_boolmask']
        ]

    for step in range(steps):
        conf_pred, pclass_pred, loss_coords, _ , summary_result, \
            precision, recall, gt_box_count, correct_hit_count, incorrect_hit_count, \
            loss, iou, valid_iou_boolmask \
            = sess.run(fetches,feed_dict=feed_dict)

        writer.add_summary(summary_result,global_step=step)
        # print('coord_preds', coord_pred)
        pprint.pprint('step={} loss={}, precision={}, recall={}, gt_box_count={}, correct_hit_count={}, incorrect_hit_count={}'.format(
            step,loss,precision,recall, gt_box_count, 
            correct_hit_count, incorrect_hit_count))
    

    # after all the steps, save to ckpt
    save_path = saver.save(sess,SAVE_PATH)
    print("model saved to {}".format(save_path))



print("end of code")
from create_net import create_training_net
import tensorflow as tf
import numpy as np
from inputloader import InputLoader
import os
import pprint
import sys
import traceback
from controlparamloader import ControlParamLoader
import shutil
from util import pred_processor, Box
import cv2
from util.Metric import Metric


# load the input and GT


def do_param_log(param_log_fd, weights, learn_r, step):
    towrite = "step={}, weights={}, lr={}\n".format(step, weights, learn_r)
    param_log_fd.write(towrite)


np_array_save_file = 'NP_SAVE.txt'
infer_output_dir = "infer_output"

# input_batch
# gt_batch
STEP_NUM = 1000000
# STEP_NUM = 1

istraining = False
debug_train_single_input = False

one_epoch_test = False

attempt_num = 17


# print out current starting configuration

print("=== CURRENT SETUP ====")
if istraining:
    task_type = "train"
else:
    task_type = "inference"

print("task type: {}".format(task_type))

if istraining:
    print("training with single sample: ", debug_train_single_input)
    print("attempt id: {}".format(attempt_num))
else:

    if one_epoch_test:
        print("infering entire dataset")
    else:
        print("infering single testcase")


user_input = input("\nwould like to proceed with current settings? (y/n)")
print(user_input)
if user_input == 'y' or user_input == 'Y' or user_input == "":
    pass
elif user_input == 'n' or user_input == 'N':
    sys.exit(0)
else:
    print("invalid input")
    sys.exit(0)


# check output dirs


pred_npz_save_dirname = "att_{}".format(attempt_num)
pred_npz_save_basedir = os.path.join(os.getcwd(), "pred_saves")
pred_npz_save_dirpath = os.path.join(
    pred_npz_save_basedir, pred_npz_save_dirname)


# setup npz file output

if os.path.exists(pred_npz_save_dirpath):
    # do nothing for now.
    # print("ignoring duplicate existence")

    reset_dir = True

    if reset_dir:
        shutil.rmtree(pred_npz_save_dirpath)
        os.mkdir(pred_npz_save_dirpath)
    else:
        raise Exception("pred_npz_save_dirpath exists")

else:
    os.mkdir(pred_npz_save_dirpath)


# setup model ckpt output
attempt_num_padded = "{}".format(attempt_num)
SAVE_PATH = "./ckpt/model-{}.ckpt".format(attempt_num_padded)


# setup param_log dir, only when training

if istraining:
    param_log_dirname = "param_log"
    param_log_dirpath = os.path.join(os.getcwd(), param_log_dirname)
    param_log_filename = "paramlog_{}.log".format(attempt_num)

    param_log_filepath = os.path.join(param_log_dirpath, param_log_filename)

    if os.path.exists(param_log_filepath):
        os.remove(param_log_filepath)

    param_log_fd = open(param_log_filepath, 'w')


if istraining:
    if debug_train_single_input:
        g1, notable_tensors, input_holders = create_training_net(
            debug_train_single_input=True)
    else:
        g1, notable_tensors, input_holders = create_training_net()
else:
    g1, notable_tensors, input_holders = create_training_net(istraining=False)


if istraining:

    if debug_train_single_input:
        inputloader = InputLoader(testcase=0)
    else:
        inputloader = InputLoader(batch_num=4)

    image_input, gt, _, essence, _ = inputloader.get_image_and_gt()

    testcase_inputloader = InputLoader(testcase=0)

    controlparamloader = ControlParamLoader()

else:
    # check if infer output dir exists or not
    infer_output_dirpath = os.path.join(os.getcwd(), infer_output_dir)
    if not os.path.exists(infer_output_dirpath):
        os.makedirs(infer_output_dirpath)

    if one_epoch_test:
        inputloader = InputLoader(batch_num=4)
        predprocessor = pred_processor.PredictionProcessor_v1()
    else:
        inputloader = InputLoader(testcase=0)
        image_input, gt, _, essence, _ = inputloader.get_image_and_gt()
        predprocessor = pred_processor.PredictionProcessor_v1()

ap_list = inputloader.get_ap_list()


# === setup gpu configuration
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.3, allow_growth=False)
config = tf.ConfigProto(gpu_options=gpu_options)

# config.gpu_options.per_process_gpu_memory_fraction = 0.5


with tf.Session(graph=g1, config=config) as sess:

    if istraining:

        single_train_run = False

        if single_train_run:
            print("training_debug True!!!!")

        writer = tf.summary.FileWriter(logdir="./summary", graph=sess.graph)
        writer.flush()

    # wanted_outputs={
    #     'net_out': g1.get_tensor_by_name("net_out"),
    #     'coord_pred' : g1.get_operation_by_name("coord_pred")
    # }

    steps = STEP_NUM

    # print('essence:',essence)
    # print('essence to pass on to:', essence[0][0])

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    # if ckpt exist, then load from it

    # saver.restore(sess, SAVE_PATH)

    # last_checkpoint = tf.train.latest_checkpoint('./ckpt/')
    # print("last_checkpoint=", last_checkpoint)
    last_checkpoint = "./ckpt/model-17.ckpt-98000"
    if not istraining:
        uinput = input("proceed with ckpt: {} ?\n".format(last_checkpoint))

        if uinput == 'y' or uinput == 'Y' or uinput == "":
            pass
        else:
            raise Exception("abort")

    if last_checkpoint is not None:
        saver.restore(sess, last_checkpoint)
        print("!!!!! restoring...!!! ")

    try:

        if istraining:

            weights = None
            learn_r = None

            for step in range(steps):

                image_input, gt, _, essence, _ = inputloader.get_image_and_gt()

                prev_weights = weights
                prev_learn_r = learn_r
                weights, learn_r = controlparamloader.getconfig()

                print("weights", weights)
                print("learn_r", learn_r)

                # check and log param is necessary
                if weights == None or learn_r == None:
                    do_param_log(param_log_fd, weights, learn_r, step)
                elif prev_weights != weights or prev_learn_r != learn_r:
                    do_param_log(param_log_fd, weights, learn_r, step)

                np_weights = np.array(weights, dtype=float)
                np_learn_r = np.array(learn_r, dtype=float)

                print("weights = ", np_weights)
                print("learn_r=", np_learn_r)

                feed_dict = {
                    input_holders['input_layer']: image_input,
                    input_holders['ground_truth']: gt,
                    input_holders['ap_list']: ap_list,
                    input_holders['essence']: essence[0][0],
                    input_holders['loss_weights']: np_weights,
                    input_holders['learning_rate']: np_learn_r
                }

                fetches = [notable_tensors['conf_pred'],
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
                           # notable_tensors['valid_iou_boolmask'],
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
                           notable_tensors['poi_iou_rawform'],
                           # notable_tensors['over_threshold_pred_conf_count']

                           ]

                pred_conf, loss_coords, _, summary_result, \
                    precision, recall, gt_box_count, correct_hit_count, incorrect_hit_count, \
                    loss, iou, gt_bbx_grid_index, gt_bbx_box_index, gt_bbx_coords, debug_gtbbx_iou, debug_pred_normalized_cxy, debug_pred_after_ap_normalized_wh, gt_mask_true_count, debug_gt_poi_conf, early_gt_poi_array, \
                    check1_poi_conf, check2_poi_conf, \
                    check_poi_pred_w, check_poi_pred_h,\
                    check_poi_gt_w, check_poi_gt_h, total_loss, \
                    debug_pred_raw_poi_w, debug_pred_raw_poi_h,\
                    debug_pred_after_exp_poi_w, debug_pred_after_exp_poi_h, loss_cxy_poi, pred_conf_poi, gt_conf_poi, gt_mask, poi_iou, poi_iou_rawform \
                    = sess.run(fetches, feed_dict=feed_dict)

                writer.add_summary(summary_result, global_step=step)

                print("STEP={}".format(step))
                # pprint.pprint('step={} loss={}, precision={}, recall={}, gt_box_count={}, correct_hit_count={}, incorrect_hit_count={}'.format(
                #     step,loss,precision,recall, gt_box_count,
                #     correct_hit_count, incorrect_hit_count))

                # np.savez(np_array_save_file,gt_mask = gt_mask, iou = iou, poi_iou = poi_iou, poi_iou_rawform = poi_iou_rawform)
                # print("gt_mask saved")

                # after all the steps, save to ckpt
                if step % 100 == 0:
                    print("FLOWING TESTECASE AT CHECKPOINT STEP #{}".format(step))

                    # flow the testcase

                    testcase_image_input, test_gt, _, essence, _ = testcase_inputloader.get_image_and_gt()

                    feed_dict = {
                        input_holders['input_layer']: testcase_image_input,
                        input_holders['ground_truth']: test_gt,
                        input_holders['ap_list']: ap_list,
                        input_holders['essence']: essence[0][0]
                    }

                    fetches = [
                        notable_tensors['pred_out_cxy'],
                        notable_tensors['pred_out_rwh'],
                        notable_tensors['pred_out_conf']
                    ]

                    pred_out_cxy, pred_out_rwh, pred_out_conf = sess.run(
                        fetches, feed_dict=feed_dict)

                    print("pred_out_cxy shape=", pred_out_cxy.shape)
                    print("pred_out_rwh shape=", pred_out_rwh.shape)
                    print("pred_out_conf shape=", pred_out_conf.shape)

                    predsave_filename = "pred_save_{:06d}".format(step)
                    predsave_filepath = os.path.join(
                        pred_npz_save_dirpath, predsave_filename)

                    np.savez(predsave_filepath, pred_out_cxy=pred_out_cxy,
                             pred_out_rwh=pred_out_rwh, pred_out_conf=pred_out_conf)
                    print("TEST FLOW DONE")

                if step % 1000 == 0:
                    # interval to save ckpt

                    # will not save ckpt if we are in training_debug
                    if not single_train_run:
                        save_path = saver.save(
                            sess, SAVE_PATH, global_step=step)
                        print("model saved to {}".format(save_path))

                if single_train_run:
                    # in training_debug, we only do a single loop of training
                    break

            print("train looping finished")

        else:
            # inferencing. not training.

            output_image_count = 0
            while True:

                image_input, gt, epoch_end_signal, essence, picked_files = inputloader.get_image_and_gt()

                if one_epoch_test and epoch_end_signal:
                    print("one epoch finished")
                    break

                feed_dict = {
                    input_holders['input_layer']: image_input,
                    input_holders['ground_truth']: gt,
                    input_holders['ap_list']: ap_list,
                    input_holders['essence']: essence[0][0]
                }

                fetches = [
                    notable_tensors['pred_out_cxy'],
                    notable_tensors['pred_out_rwh'],
                    notable_tensors['pred_out_conf'],
                    notable_tensors['loss_cx'],
                    notable_tensors['loss_cy'],
                    notable_tensors['loss_rw'],
                    notable_tensors['loss_rh'],
                    notable_tensors['loss_conf'],
                ]

                pred_out_cxy, pred_out_rwh, pred_out_conf, \
                    loss_cx, loss_cy, loss_rw, loss_rh, loss_conf = sess.run(
                        fetches, feed_dict=feed_dict)

                # print("pred_out_cxy shape=", pred_out_cxy.shape)
                # print("pred_out_rwh shape=", pred_out_rwh.shape)
                # print("pred_out_conf shape=", pred_out_conf.shape)

                # print("loss_cx=",loss_cx)
                # print("loss_cy=", loss_cy)
                # print("loss_rw=", loss_rw)
                # print("loss_rh=", loss_rh)
                # print("loss_conf=", loss_conf)

                if one_epoch_test:

                    boxmanager = Box.BoxManager()

                    # do something
                    print("size of batch = ", len(image_input))
                    for item_index in range(len(image_input)):
                        single_pred_cxy = pred_out_cxy[item_index]
                        single_pred_rwh = pred_out_rwh[item_index]
                        single_pred_conf = pred_out_conf[item_index]
                        single_gt = gt[item_index]
                        single_image = image_input[item_index]

                        boxlist = boxmanager.convert_for_single_image(
                            single_pred_cxy, single_pred_rwh, single_pred_conf, applyNMS=True)
                        gtboxlist = boxmanager.get_gt_boxes(single_gt)

                        processed_image = predprocessor.draw_all_boxes(
                            boxlist, single_image, gtboxlist=gtboxlist)

                        # processed_image = predprocessor.draw_all_bbx(pred_out_cxy=single_pred_cxy,
                        #                                              pred_out_rwh=single_pred_rwh,
                        #                                              pred_out_conf=single_pred_conf,
                        #                                              gt_arr=single_gt,
                        #                                              image=single_image)

                        output_image_filename = "{:03d}.png".format(
                            output_image_count)
                        output_image_count += 1

                        cv2.imwrite(output_image_filename, processed_image)
                        shutil.move(output_image_filename, infer_output_dir)

                        # save the image and move it to output dir

                else:

                    boxmanager = Box.BoxManager()
                    gtboxlist = boxmanager.get_gt_boxes(gt[0])

                    single_pred_cxy = pred_out_cxy[0]
                    single_pred_rwh = pred_out_rwh[0]
                    single_pred_conf = pred_out_conf[0]
                    single_gt = gt[0]
                    single_image = image_input[0]

                    boxlist = boxmanager.convert_for_single_image(
                        single_pred_cxy, single_pred_rwh, single_pred_conf, applyNMS=True)

                    # boxlist_noNMS = boxmanager.convert_for_single_image(
                    #     pred_out_cxy, pred_out_rwh, pred_out_conf)

                    processed_image = predprocessor.draw_all_boxes(
                        boxlist, single_image, gtboxlist=gtboxlist)

                    cv2.imwrite("single_test_infer.png", processed_image)

                    # single_boxlist = boxlist_noNMS[0]
                    # processed_image = predprocessor.draw_all_boxes(
                    #     single_boxlist, image_input[0], gtboxlist=gtboxlist)

                    # cv2.imwrite("single_test_infer_noNMS.png", processed_image)

                    metric = Metric()
                    precision, recall = metric.eval(gtboxlist, boxlist)

                    print("precision={}, recall={}".format(precision, recall))

                    break

    except Exception as e:

        print("exception occured.")
        print(e)
        traceback.print_exc()


print("end of code")

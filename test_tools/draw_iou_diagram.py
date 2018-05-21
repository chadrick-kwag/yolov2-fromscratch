import numpy as np
import sys
import os
sys.path.append('..')
from inputloader import InputLoader

import cv2


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def convert(given_data):
    cx, cy , rw, rh = given_data
    left_bottom_corner = (cx-rw/2, (1-cy)-rh/2)
    return left_bottom_corner, rw,rh


def calc_iou(gt_data, pred_data):
    gt_cx, gt_cy, gt_rw, gt_rh = gt_data
    gt_left_top_coord = (gt_cx - gt_rw/2, gt_cy - gt_rh/2)
    gt_right_bottom_coord = (gt_cx + gt_rw/2, gt_cy + gt_rh/2)


    pred_cx , pred_cy, pred_rw, pred_rh = pred_data
    pred_left_top_coord = ( pred_cx - pred_rw/2, pred_cy - pred_rh/2)
    pred_right_bottom_coord = (pred_cx + pred_rw/2, pred_cy + pred_rh/2)


    intersection_left_top = (max(gt_left_top_coord[0], pred_left_top_coord[0]), 
    max(gt_left_top_coord[1], pred_left_top_coord[1]))

    intersection_right_bottom = ( min (pred_right_bottom_coord[0], gt_right_bottom_coord[0]),
    min(pred_right_bottom_coord[1], gt_right_bottom_coord[1] ))


    intersection_w= intersection_right_bottom[0] - intersection_left_top[0]
    intersection_h= intersection_right_bottom[1] - intersection_left_top[1]

    gt_area = gt_rw * gt_rh
    pred_area = pred_rw * pred_rh
    intersection_area = intersection_w * intersection_h

    total_area = gt_area + pred_area - intersection_area

    iou = intersection_area / total_area

    return iou


# load the gt values
il = InputLoader(testcase=1)
_, gt, _ , essence = il.get_image_and_gt()

ess = essence[0][0]

target_grid_index = ess[0] 
target_box_index = ess[1]
gt_cx = ess[2]
gt_cy = ess[3]
gt_rw = ess[4]
gt_rh = ess[5]

gt_data=(gt_cx,gt_cy,gt_rw,gt_rh)

## load the npz files

NPZFILE_DIR='../pred_saves/tt'


npzfile_list = os.listdir(NPZFILE_DIR)
npzfile_list.sort()


debug= True

# assume that all files are .npz files

for f in npzfile_list:
    # recreate the full path
    path = os.path.join(NPZFILE_DIR,f)
    npzfiles = np.load(path)

    # print(npzfiles.files)


    pred_out_cxy = npzfiles['pred_out_cxy']
    pred_out_rwh = npzfiles['pred_out_rwh']
    pred_out_conf = npzfiles['pred_out_conf']

    print("pred_out_cxy shape=", pred_out_cxy.shape)

    print("pred_out_cxy shape=", pred_out_cxy.shape)

    pred_cx = pred_out_cxy[0,target_grid_index, target_box_index,0]
    pred_cy = pred_out_cxy[0,target_grid_index, target_box_index, 1]

    pred_rw = pred_out_rwh[0, target_grid_index, target_box_index, 0]
    pred_rh = pred_out_rwh[0, target_grid_index, target_box_index, 1]

    pred_conf = pred_out_conf[0, target_grid_index, target_box_index, 0]

    pred_data = (pred_cx, pred_cy, pred_rw, pred_rh )

    left_bottom_corner, rw,rh= convert(gt_data)
    c2,rw2,rh2 = convert(pred_data)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    ax1.add_patch(
    patches.Rectangle(
        left_bottom_corner, rw,rh, fill=False
        # (0.1, 0.1),   # (x,y)
        # 0.5,          # width
        # 0.5,          # height
        )
    )

    # print(c2,rw2,rh2)
    ax1.add_patch(patches.Rectangle(
        c2,rw2,rh2, fill=False, edgecolor="red"
    ))

    ax1.autoscale(enable=True)

    iou = calc_iou(gt_data, pred_data)

    print("iou=", iou)

    

    plt.savefig('test.png')

    p1 = cv2.imread('test.png')

    print(p1.shape)

    image_h = p1.shape[0]
    image_w = p1.shape[1]

    point_x = int(image_w *0.75)
    point_y = int(image_h *0.8)


    stringtoput = "iou={:5f}".format(iou)
    cv2.putText(p1,stringtoput,(point_x,point_y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2,cv2.LINE_AA)

    cv2.imwrite('test-pp.png', p1)


    if debug:
        break



# print(npzfile_list)
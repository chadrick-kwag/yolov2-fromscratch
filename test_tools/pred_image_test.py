import numpy as np 
import os
import sys
sys.path.append('..')
from inputloader import InputLoader
import json
import cv2
import matplotlib.pyplot as plt


def get_rect_points(cx,cy,w,h):
    p1=( int(cx-w/2) ,  int(cy-h/2) )
    p2 = (  int(cx+w/2) , int(cy+ h/2)  )
    return p1,p2


il = InputLoader(testcase=1)

input_image_dir = il.images_directory
annotation_dir = il.annotation_directory

threshold = 0.5



print(input_image_dir)

image_batch, gt_batch, epoch_end_signal, essence_batch, picked_files = il.get_image_and_gt()

reduced_image = image_batch[0]

reduced_image = cv2.cvtColor(reduced_image, cv2.COLOR_RGB2BGR)

annotation_file = picked_files[0]
annotation_file = os.path.join(annotation_dir, annotation_file)



with open(annotation_file,'r') as openf:
    jsobj = json.load(openf)
    print(jsobj)

    orig_image_w = jsobj['w']
    orig_image_h = jsobj['h']


# load the pred values
npzfile_path = "../pred_save.npz"
npzfile = np.load(npzfile_path)

print(npzfile.files)

pred_out_cxy = npzfile['pred_out_cxy']
pred_out_rwh = npzfile['pred_out_rwh']
pred_out_conf = npzfile['pred_out_conf']

pred_out_cxy = pred_out_cxy[0]
pred_out_rwh = pred_out_rwh[0]
pred_out_conf = pred_out_conf[0]

print(pred_out_cxy.shape)

index_arrs = np.where(pred_out_conf> threshold )

grid_arr = index_arrs[0]
box_arr = index_arrs[1]

grid_arr = grid_arr.reshape(1, grid_arr.shape[0])
box_arr = box_arr.reshape(1, box_arr.shape[0])

assert grid_arr.shape[1] == box_arr.shape[1]

debug = False

for i in range(grid_arr.shape[1]):
    grid_index = grid_arr[0][i]
    box_index = box_arr[0][i]
    
    print('grid_index = ', grid_index)
    print('box_index = ', box_index)


    cx = pred_out_cxy[grid_index,box_index,0]
    cy = pred_out_cxy[grid_index, box_index, 1]

    rw = pred_out_rwh[grid_index,box_index, 0]
    rh = pred_out_rwh[grid_index, box_index, 1]

    grid_cell_size = 416/13

    grid_x_index = grid_index % 13
    grid_y_index = int(grid_index / 13)

    o_x = grid_x_index * grid_cell_size
    o_y = grid_y_index * grid_cell_size

    abs_cx = o_x+ cx * grid_cell_size
    abs_cy = o_y + cy* grid_cell_size

    abs_w = grid_cell_size * rw
    abs_h = grid_cell_size * rh

    p1,p2 = get_rect_points(abs_cx, abs_cy, abs_w, abs_h)

    print(p1,p2)
    

    cv2.rectangle(reduced_image,p1,p2, (0,0,255), thickness=2)

    
    if debug:
        break

cv2.imshow('image', reduced_image)
cv2.waitKey(0)
output_filename = 'testcase_1_threshold_{}.png'.format(threshold)
cv2.imwrite(output_filename, reduced_image)

print("end of code")
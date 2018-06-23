import numpy as np 
import os
import sys
sys.path.append('..')
from inputloader import InputLoader
import json
import cv2
import matplotlib.pyplot as plt
import re, shutil


def get_rect_points(cx,cy,w,h):
    p1=( int(cx-w/2) ,  int(cy-h/2) )
    p2 = (  int(cx+w/2) , int(cy+ h/2)  )
    return p1,p2


il = InputLoader(testcase=0)

input_image_dir = il.images_directory
annotation_dir = il.annotation_directory

threshold = 0.5
save_conf_histogram = True
save_image_output = True
show_bbxed_image = False

OUTPUT_DIR="analysis_image_output"

OUTPUT_DIR_FULLPATH = os.path.abspath(OUTPUT_DIR)

npzfile_dir = "../pred_saves/att_17"



if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    

os.mkdir(OUTPUT_DIR)



if not save_conf_histogram and not save_image_output:
    print("nothing to do")
    sys.exit(0)


if save_image_output:


    image_batch, gt_batch, epoch_end_signal, essence_batch, picked_files = il.get_image_and_gt()

    reduced_image_orig = image_batch[0]
    
    gt_arr = gt_batch[0]

    reduced_image_orig = cv2.cvtColor(reduced_image_orig, cv2.COLOR_RGB2BGR)

    annotation_file = picked_files[0]
    annotation_file = os.path.join(annotation_dir, annotation_file)



    with open(annotation_file,'r') as openf:
        jsobj = json.load(openf)
        # print(jsobj)

        orig_image_w = jsobj['w']
        orig_image_h = jsobj['h']


## 

# load the pred values


npzfile_list = os.listdir(npzfile_dir)
npzfile_list.sort()


debug2 = False

pred_poi_cx_list=[]
pred_poi_cy_list=[]
pred_poi_rw_list=[]
pred_poi_rh_list=[]


single_forward_test = False

for f in npzfile_list:

    
    if single_forward_test:
        m = re.match(r"pred_save.npz",f)
    else:
        m = re.match(r"pred_save_(\d+)\.npz", f)

    if m is None:
        continue

    if single_forward_test:
        step_num=1
    else:        
        step_num = m.group(1)
    print("step num={}".format(step_num))
    


    # npzfile_path = "../pred_save.npz"
    npzfile_path = os.path.join(npzfile_dir, f)
    npzfile = np.load(npzfile_path)



    pred_out_cxy = npzfile['pred_out_cxy']
    pred_out_rwh = npzfile['pred_out_rwh']
    pred_out_conf = npzfile['pred_out_conf']

    pred_out_cxy = pred_out_cxy[0]
    pred_out_rwh = pred_out_rwh[0]
    pred_out_conf = pred_out_conf[0]

    # get histogram information of conf

    pred_out_conf_spread = np.reshape(pred_out_conf, [-1])
    # print(pred_out_conf_spread.shape)

    if save_conf_histogram:
        plt.close('all')
        fig, ax = plt.subplots()
        ax.hist(pred_out_conf_spread,bins=10, range=(0.0, 1.0))
        ax.set_ylim([0,700])
        # plt.show()
        outimagefilename = "conf_histogram_{}.png".format(step_num)
        outimagepath = os.path.join(OUTPUT_DIR, outimagefilename)
        fig.savefig(outimagefilename)
        os.rename(outimagefilename, outimagepath)
        


    

    if save_image_output:

        index_arrs = np.where(pred_out_conf> threshold )

        print("over threshold conf count=", len(index_arrs[0]))

        grid_arr = index_arrs[0]
        box_arr = index_arrs[1]

        grid_arr = grid_arr.reshape(1, grid_arr.shape[0])
        box_arr = box_arr.reshape(1, box_arr.shape[0])

        assert grid_arr.shape[1] == box_arr.shape[1]

        debug = False

        # reset cv2 image
        reduced_image = reduced_image_orig.copy()

        for i in range(grid_arr.shape[1]):

            
            grid_index = grid_arr[0][i]
            box_index = box_arr[0][i]
            
            # print('grid_index = ', grid_index)
            # print('box_index = ', box_index)


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

            # print(p1,p2)
            

            cv2.rectangle(reduced_image,p1,p2, (0,0,255), thickness=2)

            
            if debug:
                break

        # draw gt box
        gt_conf = gt_arr[:,:,4]
        # print("gt_conf shape",gt_conf.shape)
        tt = np.where(gt_conf==1.0)
        # print(tt)
        gt_conf_grid_index = tt[0][0]
        gt_conf_box_index = tt[1][0]

        gt_cx = gt_arr[gt_conf_grid_index, gt_conf_box_index, 0]
        gt_cy = gt_arr[gt_conf_grid_index, gt_conf_box_index, 1]
        gt_rw = gt_arr[gt_conf_grid_index, gt_conf_box_index, 2]
        gt_rh = gt_arr[gt_conf_grid_index, gt_conf_box_index, 3]


        grid_cell_size = 416/13

        grid_x_index = gt_conf_grid_index % 13
        grid_y_index = int(gt_conf_grid_index / 13)

        o_x = grid_x_index * grid_cell_size
        o_y = grid_y_index * grid_cell_size

        abs_cx = o_x+ gt_cx * grid_cell_size
        abs_cy = o_y + gt_cy * grid_cell_size

        abs_w = grid_cell_size * gt_rw
        abs_h = grid_cell_size * gt_rh

        p1,p2 = get_rect_points(abs_cx, abs_cy, abs_w, abs_h)

        # print(p1,p2)
        

        cv2.rectangle(reduced_image,p1,p2, (0,255,0), thickness=2)



        # print("gt_conf_grid_index={}, gt_conf_box_index={}".format(gt_conf_grid_index, gt_conf_box_index))

        # print gt cxy,rwh value and pred value
        # print("gt cx={:4f}, cy={:4f}, rw={:4f}, rh={:4f}".format(gt_cx,gt_cy, gt_rw, gt_rh))

        pred_poi_cx = pred_out_cxy[gt_conf_grid_index, gt_conf_box_index,0]
        pred_poi_cy = pred_out_cxy[gt_conf_grid_index, gt_conf_box_index,1]
        pred_poi_rw = pred_out_rwh[gt_conf_grid_index, gt_conf_box_index,0]
        pred_poi_rh = pred_out_rwh[gt_conf_grid_index, gt_conf_box_index,1]

        pred_poi_cx_list.append(pred_poi_cx)
        pred_poi_cy_list.append(pred_poi_cy)
        pred_poi_rw_list.append(pred_poi_rw)
        pred_poi_rh_list.append(pred_poi_rh)

        # print("pred cx={:4f}, cy={:4f}, rw={:4f}, rh={:4f}".format(pred_poi_cx, pred_poi_cy, pred_poi_rw, pred_poi_rh))



        if show_bbxed_image:
            cv2.imshow('image', reduced_image)
            cv2.waitKey(0)
        output_filename = 'testcase_1_{}_threshold_{}.png'.format(step_num, threshold)
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        cv2.imwrite(output_filename, reduced_image)
        os.rename(output_filename, output_filepath)

    # end of if save_image_output

    if debug2:
        break
# end of npz file loop 
 

pred_poi_cx_list = np.array(pred_poi_cx_list)
pred_poi_cy_list = np.array(pred_poi_cy_list)
pred_poi_rw_list = np.array(pred_poi_rw_list)
pred_poi_rh_list = np.array(pred_poi_rh_list)

if len(pred_poi_cx_list) == 0:
    raise Exception("nothing to show in cxyrwh trajectory")
elif len(pred_poi_cx_list) ==1:
    needtoscatter = True
    dummy_x=[0]
else:
    needtoscatter = False

plt.close('all')

fig, axes = plt.subplots(2,2)
ax1 = axes[0,0]
if needtoscatter:
    ax1.scatter(x=dummy_x,y=pred_poi_cx_list)
else:
    ax1.plot(pred_poi_cx_list)
ax1.set_ylim([0.0, 1.0])
ax1.axhline(y=gt_cx, color='red')
ax1.set_title("cx")

ax2 = axes[0,1]
if needtoscatter:
    ax2.scatter(x=dummy_x, y=pred_poi_cy_list)
else:
    ax2.plot(pred_poi_cy_list)
ax2.set_ylim([0.0, 1.0])
ax2.axhline(y=gt_cy, color='red')
ax2.set_title("cy")

ax3 = axes[1,0]
if needtoscatter:
    ax3.scatter(x=dummy_x, y=pred_poi_rw_list)
else:
    ax3.plot(pred_poi_rw_list)
ylimit = max ( max(pred_poi_rw_list)*1.1 , gt_rw*1.1)
ax3.set_ylim([0.0, ylimit])
ax3.axhline(y=gt_rw, color='red')
ax3.set_title("rw")

ax4 = axes[1,1]
if needtoscatter:
    ax4.scatter(x=dummy_x,y=pred_poi_rh_list)
else:
    ax4.plot(pred_poi_rh_list)
ylimit = max( max(pred_poi_rh_list)*1.1 , gt_rh*1.1)
ax4.set_ylim([0.0, ylimit])
ax4.axhline(y=gt_rh, color='red')
ax4.set_title("rh")

plt.subplots_adjust(wspace=0.5,hspace=0.5)


testcase_cxyrwh_trajectory_filename='testcase_1_cxyrwh_trajectory.png'
fig.savefig(testcase_cxyrwh_trajectory_filename)

plt.show()
print("testcase 1 cxyrwh trajectory image file saved")




    

print("end of code")
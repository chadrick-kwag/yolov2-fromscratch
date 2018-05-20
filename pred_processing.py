import numpy as np 

npzfiles = np.load("pred_save_3000_tc2.npz")

print(npzfiles.files)

pred_out_cxy = npzfiles['pred_out_cxy']
pred_out_rwh = npzfiles['pred_out_rwh']
pred_out_conf = npzfiles['pred_out_conf']

valid_conf = np.greater_equal(pred_out_conf,0.5)

print("valid_conf shape=", valid_conf.shape)

maxval = np.max(pred_out_conf)
maxarg = np.argmax(pred_out_conf)
print("maxval = ", maxval)
print("maxarg=", maxarg)


# maxarg = 73

# grid_index = int(maxarg / 5)
# box_index = maxarg % 5

grid_index = 96
box_index = 3
print("target conf = ", pred_out_conf[0,grid_index,box_index,:])

target_conf = pred_out_conf[0,grid_index,box_index,0]

target_cxy = pred_out_cxy[0,grid_index, box_index, :]
target_rwh = pred_out_rwh[0,grid_index, box_index, :]

print("target_cxy = ", target_cxy)
print("target_rwh = ", target_rwh)

print("copy format= {:5f}, {:5f}, {:5f}, {:5f}".format(target_cxy[0], target_cxy[1], target_rwh[0], target_rwh[1]))
print("conf = {:5f}".format(target_conf))
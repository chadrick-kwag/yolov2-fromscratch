import numpy as np 

np_save_file = 'NP_SAVE.txt.npz'

npzfiles = np.load(np_save_file)

print(npzfiles.files)
arr = npzfiles['gt_mask']

iou = npzfiles['iou']
poi_iou = npzfiles['poi_iou']
poi_iou_rawform = npzfiles['poi_iou_rawform']

# print(arr)
print(np.argmax(arr))
stretched_index = np.argmax(arr)
grid_index = int(stretched_index / 5)
box_index = stretched_index % 5

print("grid_index=", grid_index)
print("box_index=", box_index)

gt_mask = arr

print("gt_mask shape=", gt_mask.shape)
print("iou shape=", iou.shape)
print("poi_iou shape=", poi_iou.shape)
print("poi_iou_rawform shape=", poi_iou_rawform.shape)

print("iou nonzero count=", np.count_nonzero(iou))

print("poi_iou_rawform nonzero count = ", np.count_nonzero(poi_iou_rawform))
print("poi_iou_rawform nonzero indice=", np.nonzero(poi_iou_rawform))

print("poi_iou val=", poi_iou)

print("poi_iou_rawform sum=", np.sum(poi_iou_rawform))


tt = iou * gt_mask
print("tt shape=", tt.shape)

print("poi_iou=", poi_iou)

print(np.count_nonzero(arr))
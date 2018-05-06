import matplotlib.pyplot as plt
import matplotlib.patches as patches

def convert(given_data):
    cx, cy , rw, rh = given_data
    left_bottom_corner = (cx-rw/2, (1-cy)-rh/2)
    return left_bottom_corner, rw,rh



fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

# input: cx, cy, rw, rh
# from this we need the leftbottom coord, and w,h

gt_data=(0.94765,0.9583333,6.5203,8.66666)
pred_data=(0.49883342,0.49976134,0.5175644,0.8269221)

left_bottom_corner, rw,rh= convert(gt_data)
c2,rw2,rh2 = convert(pred_data)

ax1.add_patch(
    patches.Rectangle(
        left_bottom_corner, rw,rh, fill=False
        # (0.1, 0.1),   # (x,y)
        # 0.5,          # width
        # 0.5,          # height
    )
)

ax1.add_patch(patches.Rectangle(
    c2,rw2,rh2, fill=False, edgecolor="red"
))

ax1.autoscale(enable=True)

# calculate the iou

#==================
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


print('gt_area', gt_area)
print('pred_area', pred_area)
print('intersection_area', intersection_area)


total_area = gt_area + pred_area - intersection_area

iou = intersection_area / total_area

print('iou', iou)


#==================


# fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')
plt.show()

# we assume grid_cell_size = 1. 
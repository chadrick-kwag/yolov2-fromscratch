import math

# 0.94765,0.9583333

def calc_cross_entropy(pred,truth):
    return -(truth*math.log(pred)+(1-truth)*math.log(1-pred))

loss_cx = calc_cross_entropy(0.9477, 0.94765)

loss_cy = calc_cross_entropy(0.9583,0.9583333 )

print(loss_cx)

print(loss_cy)

print(loss_cx+loss_cy)
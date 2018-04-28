gt processor

get the json image_w/h and rects info

convert this to net_out_format

should i create a class that does this?

or just create a function for this?

I'm going to do this in batches anyway

just create function that will do for one

...
process

# devnotes


resize the whole thing first?

iw,ih
for each rect:
calculate which grid cell index , cx,cy,bw,bh

with bw,bh -> find which AP fits best
-> for that AP only, specify value. other APs(boxes) will have just zero
-> for the selected bestbox, set conf=1.0 and pclass=1.0

so for each object, it will have one bestbox

in this case, there would be only one object(myface)
so there will be only one bestbox
[-1,HW,B,C']
C'=cx,cy,bw',bh',conf,pclass


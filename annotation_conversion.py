# assume that a proper json object has been given

def conv_to_xywh(x1,x2,y1,y2):
	# assume that size comparision has already been resolved

	cx= (x1+x2)/2
	cy= (y1+y2)/2

	w = x2-x1
	h = y2-y1

	return (cx,cy,w,h)


def find_grid_cell(image_w, image_h , cx, cy ,grid_slice_num):
	# assumes that we slice the same number in both x, y directions.

	grid_slice_x = image_w / grid_slice_num
	grid_slice_y = image_h / grid_slice_num

	x_index = (int) (cx / grid_slice_x)
	y_index = (int) (cy / grid_slice_y)

	print(x_index, y_index)

	# we return the index in the 1d array of HW

	return grid_slice_num*y_index + x_index

def convert_annotation(input_json,grid_slice_num):
	# we don't need the name of the image file
	# we do need the w,h of the image file

	image_w = input_json['w']
	image_h = input_json['h']

	print(image_w,image_h)

	# assume the grid size=7

	objects = input_json['objects']

	for obj in objects:
		rect=obj['rect']

		x1=rect['x1']
		x2=rect['x2']
		y1=rect['y1']
		y2 = rect['y2']

		# check the size comparison

		if x1>x2:
			temp = x1
			x1= x2
			x2 = temp

		if y1> y2:
			temp = y1
			y1 = y2
			y2 = temp

		cx,cy,w,h = conv_to_xywh(x1,x2,y1,y2)
		print(cx,cy,w,h)




		#assume we only have one class

		# now from the raw cx,cy,w,h, we need to locate which grid cell this belongs to.
		# this would be determined by the cx, cy.

		# locating which grid cell the bounding box belongs to.

		gridcellnum = find_grid_cell(image_w,image_h,cx,cy,grid_slice_num)

		print("gridcellnum = {}".format(gridcellnum))



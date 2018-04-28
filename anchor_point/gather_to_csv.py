"""
this gather info from all the annotation files that we have into a csv file.
the info that we are trying to gather is each boxes w&h divided by the image w&h

"""

import csv
import os,sys
import json
# gather the w,h of the jsons and turn them to csv

DATA_BASE_DIR="/home/chadrick/prj/darkflow_otherfiles/data/face_set"

CSV_SAVE_FILE="tempsave.csv"

dirlist=[]
annotdirlist=[]

dirlist=os.listdir(DATA_BASE_DIR)

print(dirlist)

for d in dirlist:
	annot_path = os.path.join(DATA_BASE_DIR,d)
	annot_path = os.path.join(annot_path,"annotations")

	if not os.path.exists(annot_path):
		print("{} doesn't exist. abort".format(annot_path))
		sys.exit(1)
	else:
		annotdirlist.append(annot_path)


# dir annotation dir existence check finished

print(annotdirlist)


csvfile = open(CSV_SAVE_FILE,'w',newline='')

csvwriter = csv.DictWriter(csvfile,fieldnames=['w','h'])

csvwriter.writeheader()


# for each annot dir, read in all the json

for d in annotdirlist:
	listjsonfiles=os.listdir(d)
	for jf in listjsonfiles:
		# check if it is a json file
		_,extname = os.path.splitext(jf)
		if not extname ==".json":
			print("file is not json file")
			continue

		jf_path = os.path.join(d,jf)
		tempjsonobj = json.load(open(jf_path))

		image_w = tempjsonobj['w']
		image_h = tempjsonobj['h']

		objects = tempjsonobj['objects']
		

		for ob in objects:
			rect = ob['rect']

			x1 = rect['x1']
			x2 = rect['x2']
			y1 = rect['y1']
			y2 = rect['y2']

			w= abs(x1-x2) / image_w
			h = abs(y1-y2) / image_h

			csvwriter.writerow({'w':w,'h':h})

	print("finished processing {}".format(d))



print("end of code")









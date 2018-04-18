# test code for annotation_conversion

import annotation_conversion
import json

FILENAME = "019_130.json"
GRID_SLICE_NUM = 7

jsonread  = json.load(open(FILENAME))

print(jsonread)

annotation_conversion.convert_annotation(jsonread, GRID_SLICE_NUM)
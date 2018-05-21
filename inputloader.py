"""
this encompasses imageloader and gtloader

"""

import os
import random
import cv2, json, numpy as np
from APManager import APManager



class InputLoader():

    search_directory='/home/chadrick/prj/darkflow_otherfiles/data/face_set/gather_all'



    annotation_directory=os.path.join(search_directory,'annotations')

    images_directory = os.path.join(search_directory,'images')

    total_files=0 # the number of annotation files

    annotation_file_list=[]

    batch_num=1

    DEBUG = False

    # should do integrity check... but I'll skip this for now.

    # I'll skip even checking if the image really exists for now...

    def __init__(self,batch_num=1,testcase=None ):

        self.batch_num = batch_num
        self.apmanager = APManager()


        if testcase == 0:
            TEST_ANNOTATION_DIR='/home/chadrick/prj/tf_practice/yolov2-fromscratch/testdataset/annotations'
            TEST_IMAGES_DIR ='/home/chadrick/prj/tf_practice/yolov2-fromscratch/testdataset/images'
            self.annotation_directory = TEST_ANNOTATION_DIR
            self.images_directory = TEST_IMAGES_DIR

        if testcase ==1 :
            TEST_ANNOTATION_DIR='/home/chadrick/prj/tf_practice/yolov2-fromscratch/testdataset/02/annotations'
            TEST_IMAGES_DIR ='/home/chadrick/prj/tf_practice/yolov2-fromscratch/testdataset/02/images'
            self.annotation_directory = TEST_ANNOTATION_DIR
            self.images_directory = TEST_IMAGES_DIR


        # return False if something is wrong
        # return True if it is okay
        # gather from the search_directory
        if not self.check_valid_datasetdir(self.search_directory):
            raise Exception("invalid dataset directory")

        self.reload_annotation_file_list()

        # check if batch_num is smaller than annotation_file_list length
        if self.batch_num > len(self.annotation_file_list):
            raise Exception("batch num is larger than number of dataset")
            
        
    def pick_annotations(self,_picknumber):
        # random pick from annotation_file_list
        picknumber = 0
        if _picknumber > len(self.annotation_file_list):
            picknumber = len(self.annotation_file_list)
        else:
            picknumber = _picknumber

        
        picked_files = self.annotation_file_list[0:picknumber]

        self.annotation_file_list = self.annotation_file_list[picknumber:]
        return picked_files
    

    def get_image_and_gt(self):
        # returns resized_image, gt, and epoch end signal

        #pick the files and update the the annotation_list with dropped annotations
        picked_files = self.pick_annotations(self.batch_num)
        epoch_end_signal = False

        # check if annotation_file_list is empty
        # if so then we have send epoch_end signal

        if not picked_files:
            epoch_end_signal = True
            self.reload_annotation_file_list()
            picked_files = self.pick_annotations(self.batch_num)

            # even after reloading and there still is a problem with fetching the picked_files, then raise exception
            if not picked_files:
                raise Exception("failed to reload annotation list and re-fetch batch_num number of annotations")

        image_batch = []
        gt_batch = []
        essence_batch = []

        for f in picked_files:
            resized_image,gt, essence= self.process_single_annotation_file(f,return_gt=True)
            gt_batch.append(gt)
            image_batch.append(resized_image)
            essence_batch.append(essence)
        
        return image_batch, gt_batch, epoch_end_signal, essence_batch, picked_files


        
    def check_valid_datasetdir(self,dirpath):
        # check if the given path contains 'images' and 'annotations' directory
        # the names should match identically
        images_path = os.path.join(dirpath,'images')
        if not os.path.exists(images_path):
            print("{} doesn't exist".format(images_path))
            return False
        
        annotations_path = os.path.join(dirpath,'annotations')
        if not os.path.exists(annotations_path):
            print("{} doesn't exist".format(annotations_path))
            return False

        return True

    def check_if_json_file(self,filepath):
        _, ext = os.path.splitext(filepath)
        if ext=='.json':
            return True
        else:
            return False

    def reload_annotation_file_list(self):

        temp_annotation_file_list = os.listdir(self.annotation_directory)

        if temp_annotation_file_list is None:
            raise Exception("no files read from annotation dir")

        temp_annotation_file_list.sort()
        
        # filtering only json files
        self.annotation_file_list = [ f for f in temp_annotation_file_list if self.check_if_json_file(f) ]

        # shuffle them
        random.shuffle(self.annotation_file_list)

        #debug
        # print("annotation_file_list length={}".format(len(self.annotation_file_list)))


    def resized_image(self,imgpath):
        """
        from given imgpath, read in the image and prepare the image array in RGB format
        """

        #check if file exists
        if not os.path.exists(imgpath):
            raise Exception("{} doesn't exist".format(imgpath))
        
        raw_array = cv2.imread(imgpath)
        rgb_format = cv2.cvtColor(raw_array,cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_format,(416,416))

        return resized


    def process_single_annotation_file(self,filepath,return_gt = False):
        """
        from one annotation file, extract the imgae input
        and prepare GT is necessary
        """

        # print("processing {}".format(filepath))

        fullpath = os.path.join(self.annotation_directory,filepath)

        fileptr = open(fullpath)
        readjsonobj = json.load(fileptr)

        imgfile = readjsonobj['imgfile']

        imgpath = os.path.join(self.images_directory,imgfile)

        resized_image = self.resized_image(imgpath)

        # prepare gt array
        if not return_gt:
            return resized_image,None

        gt_return, essence = self.process_gt_from_annotation_info(readjsonobj)
        # print('gt_return',gt_return)

        return resized_image, gt_return, essence

    

    def process_gt_from_annotation_info(self, jsonobj, grid_slice_num=13):
        """
        from the jsonobj info(iw,ih, rects), convert them to net_out_format gt
        will return in [HW,B,C] format (C=x,y,w,h+conf+p_class = 6)
        """

        # the json file should give out the x1,x2,y1,y2 values. 
        iw = jsonobj['w'] # image width
        ih = jsonobj['h'] # image height
        # print('iw',iw)
        # print('ih',ih)

        # iw_resize_factor = 416/iw
        # ih_resize_factor = 416/ih

        obj_list = jsonobj['objects']
        for single_object in obj_list:
            rect = single_object['rect']
            x1 = rect['x1']
            x2 = rect['x2']
            y1 = rect['y1']
            y2 = rect['y2']
            name = single_object['name']

            # checking x1,x2,y1,y2 size comparison
            if x1> x2:
                temp = x1
                x1 = x2
                x2 = temp

            if y1> y2:
                temp = y1
                y1 = y2
                y2 = temp

            # print('x1',x1)
            # print('x2',x2)

            
            bw = abs(x1-x2)
            bh = abs(y1-y2)

            # print('bw',bw)
            # print('bh',bh)


            single_grid_w = iw / grid_slice_num
            single_grid_h = ih / grid_slice_num

            # thw bw,bh is the box width and height by the raw dimension
            # we need to resize this to match in the 416 x 416 dimension.
            # in other words, resized_bw and resized_bh is what the 
            # predition * AP should be
            resized_bw = bw / single_grid_w
            resized_bh = bh / single_grid_h

            # now we need to find which AP fits this best
            best_B_index = self.apmanager.best_matching_ap_index(resized_bw,resized_bh)



            # find which grid index the cx,cy belongs to
            cx = (x1+x2)/2
            cy = (y1+y2)/2

            # print('cx',cx)
            # print('cy',cy)


            grid_x_index = int(cx*grid_slice_num/iw)
            grid_y_index = int(cy*grid_slice_num/ih)

            # print('grid_x_index',grid_x_index)
            # print('grid_y_index',grid_y_index)


            center_xy_grid_index = int(grid_y_index * grid_slice_num + grid_x_index)


            # caculate relative cx, cy value
            

            cxy_grid_topleft_x = single_grid_w * grid_x_index
            cxy_grid_topleft_y = single_grid_h * grid_y_index

            # print('cxy_grid_topleft_x',cxy_grid_topleft_x)
            # print('cxy_grid_topleft_y',cxy_grid_topleft_y)

            d_cx = cx - single_grid_w * grid_x_index
            d_cy = cy - single_grid_h * grid_y_index

            # print('d_cx',d_cx)
            # print('d_cy',d_cy)

            r_cx = d_cx / single_grid_w
            r_cy = d_cy / single_grid_h

            # print('r_cx',r_cx)
            # print('r_cy',r_cy)
            

            # prepare the GT array to return
            number_of_grids = grid_slice_num*grid_slice_num
            number_of_boxes = self.apmanager.number_of_AP
            number_of_predictions_for_each_box = 6
            returnarray = np.zeros(shape=(number_of_grids, number_of_boxes,number_of_predictions_for_each_box))

            # print('center_xy_grid_index',center_xy_grid_index)
            # print('best_B_index',best_B_index)
            # should be a list of shape (number_of_predictions_for_each_box,)
            # the second last 1.0 = conf
            # the very last 1.0 = P_class. there is only one class so only one 1.0 for now.
            returnarray[center_xy_grid_index,best_B_index,:] = [r_cx,r_cy,resized_bw,resized_bh,1.0,1.0]

            # print("changed part",returnarray[center_xy_grid_index,best_B_index,:])

            # print("changed row",returnarray[center_xy_grid_index,:,:])
            # print('after populating..',returnarray)

            # also return the essence
            essence = list()
            essence.append([center_xy_grid_index,best_B_index,r_cx,r_cy,resized_bw,resized_bh])

            return returnarray, essence

    def get_ap_list(self):
        return self.apmanager.get_ap_list()
            



            
        






    


    
    
        

            
        

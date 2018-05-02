import os
import csv, pandas as pd, math, numpy as np

class APManager:
    
    csvfilepath='/home/chadrick/prj/tf_practice/yolov2-fromscratch/anchor_point/anchor_points.csv'
    AP_df=None

    def __init__(self,csvfilepath=None):
        if csvfilepath is not None:
            self.csvfilepath = csvfilepath

        # check if csv file exists
        if not os.path.exists(self.csvfilepath):
            raise Exception("csvfile doesn't exist")
        
        

        csvfp = open(self.csvfilepath,'r')
        result = pd.read_csv(csvfp,header=None)

        if result is None:
            raise Exception("failed to load anything from csv file")
        
        self.AP_df = result
        self.number_of_AP = result.shape[0]

    def best_matching_ap_index(self,rw,rh):
        # returns the index of ap which best matches the given relative_w, relative_h

        distance_list = np.zeros(shape=(self.AP_df.shape[0]))

        for ap in self.AP_df.itertuples():
            index,ap_w,ap_h = ap
            # print("ap_w:{}, ap_h:{}".format(ap_w,ap_h))
            dw = math.sqrt(math.pow(rw-ap_w,2))
            dh = math.sqrt(math.pow(rh-ap_h,2))
            
            distance_list[index]= dw+dh
        

        retval = np.argmin(distance_list)
        return retval
    def get_ap_list(self):
        return self.AP_df.values

            
        

        


        




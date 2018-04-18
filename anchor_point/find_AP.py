"""
this file will try to find the APs through k-means clustering.
the input data will be the extracted 
"""

import numpy as np 
import pandas as pd 
import random
from matplotlib import pyplot as plt
from copy import deepcopy 

import csv



def random_float(low,high):
    return random.random()*(high-low)+low

def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)

CSV_SAVE_FILE="tempsave.csv"

# csvfile = open(CSV_SAVE_FILE,newline='')
# csvreader = csv.reader(csvfile)

# # skip the headers
# next(csvreader,None)

# for row in csvreader:
#     print(row)


data = pd.read_csv(CSV_SAVE_FILE)
# print(data)

xdata = data['w'].values
ydata = data['h'].values

Z = np.array(list(zip(xdata,ydata)))


# print(data[0:3])
# print(xdata[0:3])
# print(ydata[0:3])

plt.scatter(xdata,ydata,c='black')


k=3

xhighlimit = np.max(xdata)
print(xhighlimit)

yhighlimit = np.max(ydata)
print(yhighlimit)

Cx = np.random.rand(k)
Cx = Cx * np.max(xdata)

Cy = np.random.rand(k)
Cy = Cy * np.max(ydata)

C = np.array(list(zip(Cx,Cy)), dtype=np.float32)

plt.scatter(Cx,Cy,marker="*",s=200)


C_old = np.zeros(C.shape)

clusters = np.zeros(len(Z))

error = dist(C, C_old, None)

step=0
while error!=0:
    for i in range(len(Z)):
        distances = dist(Z[i],C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    
    C_old = deepcopy(C)
    
    for  i in range(k):
        points = [Z[j] for j in range(len(Z)) if clusters[j]==i]
        # print("points for C[{}]={}".format(i,points))
        print("lengt of points for {} = {}".format(i,len(points)))
        
        if len(points)!=0:
            C[i] = np.mean(points,axis=0)
        else:
            print("points for {} is empty. retaining C[{}] value".format(i,i))

        print("C[{}]={}".format(i,C[i]))

    print("updated C={}".format(C))
    
    error = dist(C,C_old,None)

    print("step={}, error={}".format(step,error))
    step+=1

print("iteration finished")
print("C={}".format(C))








plt.show()

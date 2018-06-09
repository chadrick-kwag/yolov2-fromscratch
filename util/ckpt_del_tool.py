import os, sys, re



def recreate_filepaths(modelid, dirpath, stepnum):
    basestr = "model-{}.ckpt-{}".format(modelid, stepnum)
    metafile = "{}{}".format(basestr,".meta")
    indexfile = "{}{}".format(basestr,".index")
    datafile = "{}{}".format(basestr, ".data-00000-of-00001")

    metapath = os.path.join(dirpath, metafile)
    indexpath = os.path.join(dirpath, indexfile)
    datapath = os.path.join(dirpath, datafile)

    templist = [metapath, indexpath, datapath]

    for item in templist:
        if not os.path.exists(item):
            print("{} doesn't exist".format(item))
    
    return metapath, indexpath, datapath

# check if using python3
if sys.version_info[0]<3:
    raise Exception("must use python3")

ckptdir="../ckpt"
ckptdirpath = os.path.join(os.getcwd(), ckptdir)

# check if ckptdirpath exists
if not os.path.exists(ckptdirpath):
    print("{} doesn't exist".format(ckptdirpath))
    sys.exit(1)



# go through the files and extract the step nums that are in there

allfiles = os.listdir(ckptdirpath)

pattern = re.compile("^model-(.+)\.ckpt-(\d+)\.index")

found_stepnums=[]

modelid = None

for f in allfiles:
    m = pattern.match(f)
    if m:
        if modelid is None:
            modelid = m.group(1)
            print("modelid=", modelid)        
        conv_num = int(m.group(2))
        found_stepnums.append(conv_num)



found_stepnums.sort()

print(found_stepnums)

# get user input

if not found_stepnums:
    print("no ckpt stepnums found. exit")
    sys.exit(1)
    


inps = input("ckpt range {} - {} found. please input range to delete.(ex: 1000 2000)\n".format(found_stepnums[0], found_stepnums[-1]))
inps = inps.rstrip()
inps_split = list(map(int,inps.split(' ')))

assert len(inps_split) == 2

# fail safe
inp_min= min(inps_split)
inp_max = max(inps_split)

# get the real min and max

act_min = inp_min
act_max = inp_max

# find the nearest stepnum to act_min

for index, num in enumerate(found_stepnums):
    if num > act_min:
        del_start_num = num
        del_start_num_index = index
        break

i=0
for i in range(len(found_stepnums)):
    # search from the end
    search_index = len(found_stepnums)-1-i
    search_stepnum = found_stepnums[search_index]

    if search_stepnum < act_max:
        break

del_end_num_index = len(found_stepnums)-1 - i
del_end_num = found_stepnums[ del_end_num_index ]

if del_end_num > act_max:
    del_end_num = 0


# compare del_start_num and del_end_num
if del_start_num > del_end_num:
    print("cannot find any stepnum that is in the given range. abort")
    sys.exit(1)


print("result range: {} - {}".format(del_start_num, del_end_num))
userinput = input("proceed? (y/n") 


if userinput == 'y' or userinput == 'Y' or userinput=="":
    pass
else:
    print("incorrect input. abort")
    sys.exit(1)

pulled_poi_steplist = found_stepnums[del_start_num_index: del_end_num_index+1]


for step in pulled_poi_steplist:
    # recreate del file names

    metapath, indexpath, datapath = recreate_filepaths(modelid,ckptdirpath,step)

    os.remove(metapath)
    os.remove(indexpath)
    os.remove(datapath)



print("job done :)")
import os, re, sys, shutil


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



CKPT_DIR="../ckpt"
CKPT_SAVE_DIR="../ckpt_storage/attempt_016"

# check if dirs exist

ckpt_dir_path = os.path.join(os.getcwd(), CKPT_DIR)
ckpt_save_dir_path = os.path.join(os.getcwd(), CKPT_SAVE_DIR)

if not os.path.exists(ckpt_dir_path):
    raise Exception("{} doesn't exist".format(ckpt_dir_path))

if not os.path.exists(ckpt_save_dir_path):
    raise Exception("{} doesn't exist".format(ckpt_save_dir_path))


allfiles = os.listdir(ckpt_dir_path)

pattern = re.compile("^model-(.+)\.ckpt-(\d+)\.index")

found_stepnums=[]

modelid = None

for f in allfiles:
    m = pattern.match(f)
    if m:
        if modelid is None:
            modelid = m.group(1)
        conv_num = int(m.group(2))
        found_stepnums.append(conv_num)



found_stepnums.sort()


inp_range = input("specify the target range (ex: 1000 2000)\n")

inp_range = inp_range.rstrip()
inp_list = list(map(int, inp_range.split(' ')))

# find the stepnum that fall into the given range

selected_steps=[]

if len(inp_list)==1:
    search_num = inp_list[0]
    if search_num in found_stepnums:
        selected_steps.append(search_num)
else:
    range_min = min(inp_list)
    range_max = max(inp_list)

    for step in found_stepnums:
        if step >= range_min and step <= range_max:
            selected_steps.append(step)


if selected_steps:
    print("selected steps:")
    print(selected_steps)
    print("copy target dir: {}".format(os.path.dirname(ckpt_save_dir_path)))
else:
    print("no steps fitting user input range found. abort")
    sys.exit(0)
    

# proceeding with copy

user_inp = input("proceed? (y/n)")

if user_inp == 'y' or user_inp == "" or user_inp == 'Y':
    pass
else:
    print("exit")
    sys.exit(0)

for step in selected_steps:
    metapath, indexpath, datapath = recreate_filepaths(modelid, ckpt_dir_path, step)
    shutil.move(metapath, ckpt_save_dir_path)
    shutil.move(indexpath, ckpt_save_dir_path)
    shutil.move(datapath, ckpt_save_dir_path)

print("job done :)")

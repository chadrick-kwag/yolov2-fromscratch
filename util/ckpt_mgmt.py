import os, shutil, sys, re
sys.path.append(os.path.dirname(__file__))
import ckpt_del_tool

if sys.version_info[0]<3:
    raise Exception("must use python3")

ckptdir="../ckpt"
ckptsavedir="../ckpt_storage"

ckptdir_path = os.path.join(os.getcwd(), ckptdir)
ckptsavedir_path = os.path.join(os.getcwd(), ckptsavedir)

if not os.path.exists(ckptdir_path):
    raise Exception("{} not found".format(ckptdir_path))

if not os.path.exists(ckptsavedir_path):
    raise Exception("{} not found".format(ckptsavedir_path))



def print_help():
    print("==============")
    print("1) empty out ckpt dir")
    print("2) copy stored ckpt to ckpt dir")
    print("==============")

def empty_ckpt_dir():
    shutil.rmtree(ckptdir_path)
    os.makedirs(ckptdir_path)

def create_checkpoint_file(ckptdirpath, modelid, stepnum):

    modelfullname = "model-{}.ckpt-{}".format(modelid, stepnum)

    checkpoint_filepath = os.path.join(ckptdirpath, "checkpoint")
    with open(checkpoint_filepath,'w') as fd:
        fd.write("model_checkpoint_path: \"{}\"\n".format(modelfullname))
        fd.write("all_model_checkpoint_paths: \"{}\"".format(modelfullname))
        fd.flush()
    
    # model_checkpoint_path: "model-16.ckpt-329000"
# all_model_checkpoint_paths: "model-16.ckpt-329000"



def attempt_copy_ckptfiles():
    # find all dirs in ckptsavedir
    dirlist = [ f for f in os.listdir(ckptsavedir_path) if os.path.isdir(os.path.join(ckptsavedir_path,f)) ]
    if len(dirlist)==0:
        raise Exception("no dirs in ckpt save dir")
    
    print("possible dirs:")
    for d in dirlist:
        print("> {}".format(d))
    
    uinput = input("which one to look into?\n")

    uinput = uinput.rstrip()
    if uinput not in dirlist:
        raise Exception("invalid input")
    
    # find the numbers available
    selected_store_dirpath = os.path.join(ckptsavedir_path,uinput)

    allfiles = os.listdir(selected_store_dirpath)

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

    if len(found_stepnums)==0:
        raise Exception("no ckpt files found")
    
    found_stepnums.sort()

    print("available steps:")

    for s in found_stepnums:
        print("> {}".format(s))

    uinput = input("select which step num to load into ckpt dir\n")    
    selected_stepnum = int(uinput.rstrip())


    if selected_stepnum not in found_stepnums:
        raise Exception("invalid input")

    metapath, indexpath, datapath = ckpt_del_tool.recreate_filepaths(modelid,selected_store_dirpath,selected_stepnum)


    # empty the ckptdir
    empty_ckpt_dir()

    shutil.copy(metapath,ckptdir_path)
    shutil.copy(indexpath, ckptdir_path)
    shutil.copy(datapath, ckptdir_path)

    # create checkpointfile?
    
    create_checkpoint_file(ckptdir_path, modelid, selected_stepnum)

    print("job done:)")

    

print_help()

uinput = input()

uinput = uinput.rstrip()

if uinput=='1':
    empty_ckpt_dir()
elif uinput=='2':
    attempt_copy_ckptfiles()
    

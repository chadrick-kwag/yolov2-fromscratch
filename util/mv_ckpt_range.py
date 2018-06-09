import os

CKPT_DIR="../ckpt"
CKPT_SAVE_DIR="../ckpt_storage/attempt_016"

# check if dirs exist

ckpt_dir_path = os.path.join(os.getcwd(), CKPT_DIR)
ckpt_save_dir_path = os.path.join(os.getcwd(), CKPT_SAVE_DIR)

if not os.path.exists(ckpt_dir_path):
    raise Exception("{} doesn't exist".format(ckpt_dir_path))

if not os.path.exists(ckpt_save_dir_path):
    raise Exception("{} doesn't exist".format(ckpt_save_dir_path))



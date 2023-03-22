

# from PIL import Image
import os
import glob
import numpy as np
import random
import shutil
random.seed(0)
SRC_NOISE = "data/external/combine_noise_sub"
SRC_CLEAN = "data/external/syn_subsidence"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
NUM_TRAIN = 10000#,2000,5000,8000, 10000 
NUM_VAL = NUM_TRAIN//4
NUM_TEST = 1000

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR,f"clean_{NUM_TRAIN+NUM_VAL}"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR,f"noisy_{NUM_TRAIN+NUM_VAL}"), exist_ok=True)

os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(os.path.join(VAL_DIR,f"clean_{NUM_TRAIN+NUM_VAL}"), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR,f"noisy_{NUM_TRAIN+NUM_VAL}"), exist_ok=True)

os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(os.path.join(TEST_DIR,"clean"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR,"noisy"), exist_ok=True)

def train_test_split(src_noise,src_clean):
    src_noise_list = glob.glob(os.path.join(src_noise,"*.png"))
    number_of_total_pairs = len(src_noise_list)

    for noisy_img in src_noise_list[:NUM_TRAIN]:
        shutil.copy(noisy_img,os.path.join(TRAIN_DIR,f"noisy_{NUM_TRAIN+NUM_VAL}",os.path.basename(noisy_img)))
        clean_img = noisy_img.replace(SRC_NOISE,SRC_CLEAN)
        shutil.copy(clean_img,os.path.join(TRAIN_DIR,f"clean_{NUM_TRAIN+NUM_VAL}",os.path.basename(clean_img)))

    for noisy_img in src_noise_list[NUM_TRAIN:NUM_TRAIN+NUM_VAL]:
        shutil.copy(noisy_img,os.path.join(VAL_DIR,f"noisy_{NUM_TRAIN+NUM_VAL}",os.path.basename(noisy_img)))
        clean_img = noisy_img.replace(SRC_NOISE,SRC_CLEAN)
        shutil.copy(clean_img,os.path.join(VAL_DIR,f"clean_{NUM_TRAIN+NUM_VAL}",os.path.basename(clean_img)))
    
    for noisy_img in src_noise_list[-NUM_TEST:]:
        shutil.copy(noisy_img,os.path.join(TEST_DIR,"noisy",os.path.basename(noisy_img)))
        clean_img = noisy_img.replace(SRC_NOISE,SRC_CLEAN)
        shutil.copy(clean_img,os.path.join(TEST_DIR,"clean",os.path.basename(clean_img)))

if __name__ == "__main__":
    train_test_split(SRC_NOISE ,SRC_CLEAN)

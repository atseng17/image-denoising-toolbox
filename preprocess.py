

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
    # print(src_noise_list[0])
    random.shuffle(src_noise_list)
    # print(src_noise_list[0])
    number_of_total_pairs = len(src_noise_list)
    # print(src_noise_list[:10])
    for noisy_img in src_noise_list[:int(number_of_total_pairs*0.6)]:
        shutil.copy(noisy_img,os.path.join(TRAIN_DIR,"noisy",os.path.basename(noisy_img)))
        clean_img = noisy_img.replace(SRC_NOISE,SRC_CLEAN)
        shutil.copy(clean_img,os.path.join(TRAIN_DIR,"clean",os.path.basename(clean_img)))

    for noisy_img in src_noise_list[int(number_of_total_pairs*0.6):int(number_of_total_pairs*0.8)]:
        shutil.copy(noisy_img,os.path.join(VAL_DIR,"noisy",os.path.basename(noisy_img)))
        clean_img = noisy_img.replace(SRC_NOISE,SRC_CLEAN)
        shutil.copy(clean_img,os.path.join(VAL_DIR,"clean",os.path.basename(clean_img)))
    
    for noisy_img in src_noise_list[int(number_of_total_pairs*0.8):]:
        shutil.copy(noisy_img,os.path.join(TEST_DIR,"noisy",os.path.basename(noisy_img)))
        clean_img = noisy_img.replace(SRC_NOISE,SRC_CLEAN)
        shutil.copy(clean_img,os.path.join(TEST_DIR,"clean",os.path.basename(clean_img)))

if __name__ == "__main__":
    train_test_split(SRC_NOISE ,SRC_CLEAN)

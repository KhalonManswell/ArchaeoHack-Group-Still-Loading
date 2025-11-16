TRAIN_RATIO = 0.8
AUGMENTATIONS_PER_TRAIN = 15

PROCESSED_PATH = r'/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/archaeohack/processed_data'
DATA_PATH = r'/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/archaeohack/data/utf-pngs'

import cv2
import os
import random
import shutil
import numpy as np


def split_data(data_path):
    """List files in `data_path` and return the list.

    This function prints a short summary rather than dumping the whole list
    so output remains readable in terminals.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    
    # Skip hidden directories (starting with dot)
    print(f"Listing files in: {data_path}")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.startswith('.'):
                continue

            filename = file
            label = os.path.splitext(filename)[0] 
            print(f"Found file: {filename} → class: {label}")

            src_path = os.path.join(root, file)
            img = cv2.imread(src_path,cv2.IMREAD_GRAYSCALE)

            # If cv2.imread failed, img will be None
            if img is None:
                print(f"Warning: failed to read image: {src_path} (skipping)")
                continue
            
            # split into train and val sets
            random_num = random.random()
            print(f"RNGesus: {random_num}")
            if random_num < TRAIN_RATIO:
                subset = 'train'
            else:
                subset = 'val'




    SOURCE_DIR = "/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/archaeohack/data/utf-pngs"
    PROCESSED_PATH = "/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/archaeohack/processed_data"

    train_root = os.path.join(PROCESSED_PATH, "train")
    val_root   = os.path.join(PROCESSED_PATH, "val")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

# List all png files
    images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".png")]

    for filename in images:
        label = os.path.splitext(filename)[0]  # "A1.png" -> "A1"

        # Create class folders
        train_class_dir = os.path.join(train_root, label)
        val_class_dir   = os.path.join(val_root, label)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        img_path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        # ------------------------
        # 1 image → validation
        # ------------------------
        if img is not None:
            val_out_dir = os.path.join(val_root, label)
            os.makedirs(val_out_dir, exist_ok=True)   # create folder **here**
    
            val_img = augment_pic(img.copy())
            val_out_path = os.path.join(val_out_dir, f"{label}_png.png")
            cv2.imwrite(val_out_path, val_img)

        # ------------------------
        # remaining images → train (augment 15x)
        # ------------------------
        if img is not None:
            train_out_dir = os.path.join(train_root, label)
            os.makedirs(train_out_dir, exist_ok=True)  # create folder **here**
    
            for i in range(AUGMENTATIONS_PER_TRAIN):
                aug_img = augment_pic(img.copy())
                train_out_path = os.path.join(train_out_dir, f"aug_{i}_{filename}")
                cv2.imwrite(train_out_path, aug_img)







"""""
            out_path = os.path.join(out_dir, filename)
            try:
                ok = cv2.imwrite(out_path, augimg)
                if subset == 'train':
                    out_path2 = os.path.join(out_dir, 'dupe_' + filename)
                    ok = cv2.imwrite(out_path2, augimg2)
            except Exception as e:
                print(f"Exception when writing {out_path}: {e}")
                ok = False

            if not ok:
                print(f"Failed to write image: {out_path}")
            else:
                print(f"Wrote image: {out_path}")
"""



    

    
    # Print a short summary so it's obvious something ran
    # print(f"Found {len(all_files)} files in: {data_path}")
    # print("First 10 entries:", all_files)
    # return all_files


def augment_pic(pic):
    # add random number of circles noise 5px diameter
    num_circles = random.randint(1, 30)
    for _ in range(num_circles):
        center_x = random.randint(0, pic.shape[1] - 1)
        center_y = random.randint(0, pic.shape[0] - 1)
        color = 0
        cv2.circle(pic, (center_x, center_y), 3, color, -1)
    # convert into binary
    _, pic = cv2.threshold(pic, 150, 255, cv2.THRESH_BINARY)
    # random erode/dilate value
    erodedilate = random.randint(-2, 2)
    if erodedilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erodedilate, erodedilate))
        transformed = cv2.dilate(pic, kernel, iterations=1)
    elif erodedilate < 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-erodedilate, -erodedilate))
        transformed = cv2.erode(pic, kernel, iterations=1)
    else:
        transformed = pic
    resized = cv2.resize(transformed, (200, 200))
    #random flip
    if random.random() < 0.5:
        pic = cv2.flip(pic, 1)
    # random shift
    vertical_shift = random.randint(-40, 40)
    horizontal_shift = random.randint(-40, 40)
    M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    pic = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]),borderValue=255)
    return pic


def main():
    print('splitting data...')
    # Use a raw string or os.path.join to avoid accidental escape sequences
    try:
        split_data(DATA_PATH)
    except Exception as e:
        print('Error while splitting data:', repr(e))
    else:
        print('data split complete.')


if __name__ == '__main__':
    main()
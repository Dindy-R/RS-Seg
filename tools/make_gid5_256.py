# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:39:54 2019

@author: Administrator
"""
import os
import cv2
import json
import random
import os.path as osp

from PIL import Image
from tqdm import tqdm
import numpy as np
from functools import partial
from multiprocessing import Pool

# GID_5_classes = {
#     'background': [0, 0, 0],
#     'built-up': [255, 0, 0],
#     'farmland': [0, 255, 0],
#     'forest': [0, 255, 255],
#     'meadow': [255, 255, 0],
#     'water': [0, 0, 255],
# }

GID_5_classes = {
    'background': [0],
    'built-up': [1],
    'farmland': [2],
    'forest': [3],
    'meadow': [4],
    'water': [5],

}

PALETTE = list(GID_5_classes.values())
# print(PALETTE)

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.uint8)
    return semantic_map


def split_single_image(img_path, target_dir, split_size, overlap):
    input_dir, img_name = os.path.split(img_path)
    img_id = img_name.rstrip('.tif')
    print(f'> {img_path}')

    stride = split_size - overlap
    img_path = os.path.join(input_dir, img_name)
    label_path = os.path.join(input_dir.rstrip('images') + 'labels', img_id + '_5label.tif')
    image = cv2.imread(img_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # bgr_label = cv2.imread(label_path)
    # rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)
    # onehot = mask_to_onehot(label, PALETTE)
    # mask = np.argmax(onehot, axis=2).astype(np.uint8)

    # os.makedirs(os.path.join(target_dir, 'mask_label'), exist_ok=True)
    # cv2.imwrite(os.path.join(target_dir, 'mask_label', img_name), mask)
    img_h, img_w, img_c = image.shape
    n_h = int(np.ceil((img_h - split_size) / stride)) + 1
    n_w = int(np.ceil((img_w - split_size) / stride)) + 1

    boxes_y1y2x1x2 = []
    for i in range(n_h):
        dh = min(i*stride, img_h-split_size)
        for j in range(n_w):
            dw = min(j*stride, img_w-split_size)
            boxes_y1y2x1x2.append([dh, dh+split_size, dw, dw+split_size])

    for box in boxes_y1y2x1x2:
        y1, y2, x1, x2 = box
        img_crop = image[y1:y2, x1:x2]
        label_crop = label[y1:y2, x1:x2]
        fname = img_id + '_' + str(y1) + '_' + str(x1) + '.tif'
        cv2.imwrite(os.path.join(target_dir, 'train', 'images', fname), img_crop)
        cv2.imwrite(os.path.join(target_dir, 'train', 'labels', fname), label_crop)

    # for box in val_set:
    #     y1, y2, x1, x2 = box
    #     img_crop = image[y1:y2, x1:x2]
    #     label_crop = mask[y1:y2, x1:x2]
    #     fname = img_id + '_' + str(y1) + '_' + str(x1) + '.tif'
    #     cv2.imwrite(os.path.join(target_dir, 'val', 'images', fname), img_crop)
    #     cv2.imwrite(os.path.join(target_dir, 'val', 'labels', fname), label_crop)


def split_images_in_folder(input_folder, target_dir, split_size, overlap):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
     # Iterate over each image file
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        img_id = img_name.rstrip('.tif')
        print(f'> {img_path}')
        stride = split_size - overlap
        label_path = os.path.join(input_folder.rstrip('images') + 'labels', img_id + '.tif')
        image = cv2.imread(img_path)
        label = Image.open(label_path)
        onehot = mask_to_onehot(label, PALETTE)
        mask = np.argmax(onehot, axis=2).astype(np.uint8)
        img_h, img_w, img_c = image.shape
        n_h = int(np.ceil((img_h - split_size) / stride)) + 1
        n_w = int(np.ceil((img_w - split_size) / stride)) + 1
        boxes_y1y2x1x2 = []
        for i in range(n_h):
            dh = min(i*stride, img_h-split_size)
            for j in range(n_w):
                dw = min(j*stride, img_w-split_size)
                boxes_y1y2x1x2.append([dh, dh+split_size, dw, dw+split_size])
        for box in boxes_y1y2x1x2:
            y1, y2, x1, x2 = box
            img_crop = image[y1:y2, x1:x2]
            label_crop = mask[y1:y2, x1:x2]
            fname = img_id + '_' + str(y1) + '_' + str(x1) + '.tif'
            cv2.imwrite(os.path.join(target_dir, 'train', 'images', fname), img_crop)
            cv2.imwrite(os.path.join(target_dir, 'train', 'labels', fname), label_crop)


if __name__ == '__main__':
    img_dir = 'D:\\BaiduNetdiskDownload\\GID\\Classification'
    target_dir = 'D:\\BaiduNetdiskDownload\\GID\\GG'
    split_size = 256
    overlap = 0

    _path = osp.join(target_dir, 'train', 'images')
    if not osp.exists(_path):
        os.makedirs(_path)

    _path = osp.join(target_dir, 'train', 'labels')
    if not osp.exists(_path):
        os.makedirs(_path)

    imgs = sorted([osp.join(img_dir, 'images', f) for f in os.listdir(osp.join(img_dir, 'images')) if f.endswith('.tif')])
    for img_path in tqdm(imgs):
        split_single_image(img_path, target_dir, split_size, overlap)
        # split_images_in_folder(img_dir, target_dir, split_size, overlap)
    
    
        
    




import cv2
import pdb
import os
import torch
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensorV2
import torch.utils.data as data
from segmentation_models_pytorch import Unet
from pathlib import Path


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, img_path, mean, std):
        # self.root = root
        # df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = img_path
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, img_path):
        # fname = self.fnames[idx]
        # path = os.path.join(self.root, fname)
        img_path = self.fnames
        # print("img_path: ", img_path)
        fname = img_path
        img_pth = "static/images/" + img_path
        image = cv2.imread(img_pth)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def take_name_mask(arrs):
    """
    The function takes the arr of 3 values from the prediction.
    Return name, mask of pixel values
    """
    cls_ids = []
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for arr in arrs:
        img_name = arr[0]
        cls_id, label = arr[2], arr[1]

        mask_label = np.zeros(256*1600, dtype=np.uint8)
        label = label.split(" ")
        position = map(int, label[0::2])
        length = map(int, label[1::2])
        for pos, le in zip(position, length):
            mask_label[pos-1:pos+le-1] = 1
        mask[:, :, cls_id-1] = mask_label.reshape(256, 1600, order="F")

        cls_ids.append(cls_id)

    return img_name, mask, cls_ids

def show_mask_image(arrs):
    """ Return the image with errors masked
    """
    
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
    train_path = Path("static/images/")

    name, mask, cls_ids = take_name_mask(arrs)
    img = cv2.imread(str(train_path / name))
    fig, ax = plt.subplots(figsize=(15,15))
    for cls_id in cls_ids:
        contours, _ = cv2.findContours(mask[:, :, cls_id-1], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, color=palet[cls_id-1], thickness=2)

    # ax.set_title(name)
    # ax.imshow(img)
    # plt.show()
    return img
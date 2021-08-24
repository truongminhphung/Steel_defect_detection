from flask import Flask, render_template, url_for, request, send_file
import cv2
import pdb
import os
import torch
# import pandas as pd
import numpy as np
from torch._C import device
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensorV2
import torch.utils.data as data
from segmentation_models_pytorch import Unet
from utilities import TestDataset, mask2rle, post_process, show_mask_image
from PIL import Image
import io
from base64 import b64encode

model_path = "model.pth"

app = Flask(__name__)
app.secret_key = "phung"

@app.route('/')
@app.route("/index")
def index():
    return render_template('index.html')
@app.route('/info')
def info():
    return render_template('information.html')

@app.route("/detect", methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        result_arr = []
        img_url = request.form["img1"]
        result_arr = run_detect(img_url)
        added_img_url = "images/" + img_url
        if len(result_arr) == 0:
            return render_template("result_normal.html", img_url=added_img_url)
        else:
            result_img = show_mask_image(result_arr)

            file_object = io.BytesIO()
            img= Image.fromarray(result_img.astype('uint8'))
            img.save(file_object, 'PNG')
            base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
        return render_template("result_defect.html", img_url = base64img)
    else:
        return render_template('index.html')


def run_detect(img_url):
    testset, best_threshold, min_size = initialize_dataloader(img_url)
    model = initialize_load_model()

    predictions = []
    count_break = 0
    for i, batch in enumerate(testset):
        fnames, images = batch
        batch_preds = torch.sigmoid(model(images))
        batch_preds = batch_preds.detach().cpu().numpy()
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                pred, num = post_process(pred, best_threshold, min_size)
                rle = mask2rle(pred)
                if len(rle) == 0:
                    continue
                name = fname
                cls_id = cls + 1
                if count_break < 1:
                    predictions.append([name, rle, cls_id])
            count_break += 1
            break
        break

    return predictions


def initialize_dataloader(img_url):
    # initialize test dataloader
    best_threshold = 0.4
    num_workers = 4
    batch_size = 1
    print('best_threshold', best_threshold)
    min_size = 3500
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    testset = DataLoader(
        TestDataset(img_url, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return testset, best_threshold, min_size


def initialize_load_model():
    """ Run model without GPU, if you want to use GPU juse uncomment device variable"""
    # device = torch.device("cuda")
    model = Unet("resnet18", encoder_weights=None, classes=4, activation=None)
    # model.to(device)
    model.eval()
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    return model


if __name__ == '__main__':
    app.run(debug=True)

import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as VT
import torch
import matplotlib.pyplot as plt
from PIL import Image
import train_utils.transforms as T
import math
import time

from speedy_orientation_util import segment_and_fix_image_range
from speedy_detection_util import showbox_no_bottomY
from speedy_crop_util import digit_segmentation
from speedy_pebble_util import updatePebbleLocation
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


# for carved testing
filenames = list(
    sorted(os.listdir('./Carved Pebbles/')))
# create directory to save processed images
vis_tgt_path = "./Carved Pebbles Predictions/"
if not os.path.isdir(vis_tgt_path):
    os.mkdir(vis_tgt_path)
# read through each image and predict
for filename in filenames:
    print(filename)
    img = cv2.imread(os.path.join(
        './Carved Pebbles/', filename))

    # downsize image
    scale_percent = 25  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # check if image has digits with confidence
    pebbleDigitsCrops, pebbleDigitBoxes, pebbleDigitScores, goodPredictions, goodMasks, originalDigitCrops = digit_segmentation(
        img)

    # see if digits were detected
    if pebbleDigitsCrops is not None:
        # save orientation bar prediction
        for i in range(len(pebbleDigitsCrops)):
            annImg, fixedImages = segment_and_fix_image_range(
                pebbleDigitsCrops[i], originalDigitCrops[i], 0.9)
            for f in range(len(fixedImages)):
                # prediciton
                predImg, predlabels, predScores = showbox_no_bottomY(
                    fixedImages[f])
                if predImg is not None:
                    cv2.imwrite(os.path.join(vis_tgt_path, str(i)+str(f)+filename), predImg)
                else:
                    cv2.putText(fixedImages[f], 'NONE', (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                4, (0, 0, 255), thickness=10)
                    cv2.imwrite(os.path.join(vis_tgt_path, str(i)+str(f)+filename), fixedImages[f])

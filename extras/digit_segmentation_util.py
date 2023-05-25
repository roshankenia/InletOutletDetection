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
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
digit_segmentation_model = torch.load(
    r'./saved_models/group_digit_detector.pkl')
digit_segmentation_model.to(device)
digit_segmentation_model.eval()


def normalize(arr):
    """
    Linear normalization
    normalize the input array value into [0, 1]
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # print("...arr shape", arr.shape)
    # print("arr shape: ", arr.shape)
    for i in range(3):
        minval = arr[i, :, :].min()
        maxval = arr[i, :, :].max()
        if minval != maxval:
            arr[i, :, :] -= minval
            arr[i, :, :] /= (maxval-minval)
    return arr


def digit_segmentation(img):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = digit_segmentation_model([img.to(device)])

    # print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
    img = np.array(img)  # tensor -> ndarray

    bboxes = prediction[0]['boxes'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    # print(bboxes)
    goodBBoxes = []
    # create digit crops
    digitCrops = []
    for i in range(len(scores)):
        if scores[i] >= 0.98:
            bbox = bboxes[i].astype(int)
            goodBBoxes.append(bbox)
            digits_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            digitCrops.append(digits_crop)
        else:
            # scores already sorted
            break

    # draw boxes if they exist
    if (len(goodBBoxes) != 0):
        return digitCrops, goodBBoxes
    return None, None

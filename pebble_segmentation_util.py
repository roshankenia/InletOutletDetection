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

# set to evaluation mode
pebble_segmentation_model = torch.load(
    './saved_models/mask-rcnn-pebble-with-neg.pt')
pebble_segmentation_model.eval()
CLASS_NAMES = ['__background__', 'not pebble', 'pebble']
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
pebble_segmentation_model.to(device)


def get_coloured_mask(mask, pred_c):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colour = [[0, 0, 255]]
    if pred_c == 'not pebble':
        colour = [[255, 0, 0]]
    # colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [
    #     255, 0, 255], [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colour
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


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


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def pebble_segmentation(img, confidence=0.98):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    annImg = img.copy()
    # adjust brightness of image
    # img = adjust_contrast_brightness(img, contrast=1.0, brightness=200)
    transform = VT.Compose([VT.ToTensor()])
    img = transform(img)

    # need to normalize first
    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    img = img.to(device)
    pred = pebble_segmentation_model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    if len(pred_t) == 0:
        return None, None, None
    masks = (pred[0]['masks'] > 0.5).detach().cpu().numpy()
    masks = masks.reshape(-1, *masks.shape[-2:])
    # print(pred[0]['labels'].numpy().max())
    pred_class = np.array([CLASS_NAMES[i]
                           for i in list(pred[0]['labels'].cpu().numpy())])
    pred_boxes = np.array([[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                           for i in list(pred[0]['boxes'].detach().cpu().numpy())])
    masks = masks[pred_t]
    pred_boxes = pred_boxes[pred_t]
    pred_class = pred_class[pred_t]
    make_mask_image(annImg, masks, pred_boxes, pred_class)
    return masks, pred_boxes, pred_class


pebNum = 0


def make_mask_image(img, masks, boxes, pred_cls, rect_th=2, text_size=2, text_th=2):
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i], pred_cls[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    # save frame as JPG file
    folder = f"./io_results/PebbleDetectionTest/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    global pebNum
    cv2.imwrite("./io_results/PebbleDetectionTest/image" +
                str(pebNum) + "_mask.jpg", img)
    pebNum += 1


def create_full_frame_crop(img, mask):
    mask = np.asarray(mask, dtype="uint8")
    # focus on mask pixels in img
    only_mask = cv2.bitwise_and(img, img, mask=mask)
    return only_mask


def crop_pebble(img, masks, boxes, ind):
    mask = np.asarray(masks[0], dtype="uint8")
    # obtain only the mask pixels from the image
    only_mask = cv2.bitwise_and(img, img, mask=mask)
    bbox = boxes[0]
    # crop the image to only contain the pebble
    crop = only_mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # put pebble on blank black background
    imgSize = max(crop.shape)
    background = np.zeros((imgSize, imgSize, 3), np.uint8)
    ch, cw = crop.shape[:2]

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((imgSize-ch)/2)
    xoff = round((imgSize-cw)/2)

    background[yoff:yoff+ch, xoff:xoff+cw] = crop
    # save crop as JPG file
    # cv2.imwrite("./ceramicimages/image" + str(ind) + "/crop.jpg", background)

    return background

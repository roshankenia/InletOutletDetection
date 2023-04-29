import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from PIL import Image
import math

warnings.filterwarnings('ignore')

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# set to evaluation mode
crop_orientation_model = torch.load(
    './saved_models/orientation_detector_complete.pt')
crop_orientation_model.eval()
CLASS_NAMES = ['__background__', 'bar']
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
crop_orientation_model.to(device)


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


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = [255, 0, 0]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img, confidence):
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
    img = Image.fromarray(img).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    img = img.to(device)
    pred = model([img])
    # print('prediction:', pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    # take only top prediction
    if len(pred_score) == 0:
        return None, None, None, None
    if len(pred_t) == 0:
        return None, None, None, None
    pred_t = [0]
    pred_t = pred_t[-1]
    masks = (pred[0]['masks'] > 0.5).detach().cpu().numpy()
    masks = masks.reshape(-1, *masks.shape[-2:])
    # print(masks.shape)
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i]
                  for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    return masks, pred_boxes, pred_class, pred_score


def getAngle(p1, p2, center):
    ang = math.degrees(math.atan2(
        p2[1]-center[1], p2[0]-center[0]) - math.atan2(p1[1]-center[1], p1[0]-center[0]))
    return ang + 360 if ang < 0 else ang


def decideIfHorizontal(minP, maxP):
    # we need to calculate the center of the box
    cx = (minP[0] + maxP[0])/2
    cy = (minP[1] + maxP[1])/2

    # now we need to find the middle point of the left side of the box
    leftx = maxP[0]
    lefty = cy

    # find angle
    angle = getAngle((leftx, lefty), (maxP[0], maxP[1]), (cx, cy))

    return angle, maxP, (leftx, lefty), (cx, cy)


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def segment_bar_instance(img, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
    segment_bar_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    # img = adjust_contrast_brightness(img, contrast=1, brightness=30)
    masks, boxes, pred_cls, pred_score = get_prediction(img, confidence)
    use = False
    bottomY = None
    if masks is None:
        cv2.putText(img, 'no detection', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 255), thickness=text_th)
    else:
        mask = masks[0]
        box = boxes[0]
        pred = pred_cls[0]
        score = pred_score[0]
        # check if horizontal or not
        angle, maxP, leftP, centP = decideIfHorizontal(box[0], box[1])
        # print('Angle:', angle)
        if angle < 20:
            cv2.putText(img, 'Aligned:' + str(round(angle, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), thickness=text_th)
            cv2.putText(img, str(round(score, 2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), thickness=text_th)
            # print('Horizontal found')
            use = True
            bottomY = max(box[0][1], box[1][1])
        else:
            cv2.putText(img, 'Not Aligned:' + str(round(angle, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), thickness=text_th)
            cv2.putText(img, str(round(score, 2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), thickness=text_th)
            # print('Horizontal not found')
        cv2.line(img, tuple(map(round, maxP)), tuple(
            map(round, centP)), (255, 255, 255), 10)
        cv2.line(img, tuple(map(round, centP)), tuple(
            map(round, leftP)), (255, 255, 255), 10)
        rgb_mask = get_coloured_mask(mask)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, box[0], box[1],
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred, box[0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    return img, use, bottomY


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def save_crops(digits_crop, rotations, orientationBarFolder):
    useCrops = []
    rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
                 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    # rotate each digit crop and select only those that are oriented horizontally
    for c in range(len(digits_crop)):
        digit_crop = digits_crop[c]
        for rotation in rotations:
            # rotate image
            rotImg = rotate_im(digit_crop, rotation)
            # resize
            rotImg = cv2.resize(rotImg, (500, 500))

            # rotImgCopy = rotImg.copy()
            # # increase brightness of digits
            # rotImg = brighten_digits(rotImg)

            # gamma correct
            # rotImg = gamma_correct(rotImg, 0.5)

            # contrast stretch image
            # rotImg = contrast_stretch(rotImg)

            # rotImg = sharpen_digits(rotImg)
            # predict oritentation bar on rotation
            rotImgWithAnnot, use, bottomY = segment_bar_instance(rotImg.copy())
            # save oritentation bar image
            cv2.imwrite(orientationBarFolder+"orientation_" + str(count) + "_digit_crop_" +
                        str(c) + "_rot" + str(rotation)+".jpg", rotImgWithAnnot)
            if use:
                # save both brightened and original
                # useCrops.append((rotImgCopy, bottomY, count, c, rotation))
                useCrops.append((rotImg, bottomY, count, c, rotation))
    return useCrops

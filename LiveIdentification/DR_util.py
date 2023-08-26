import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
import torch
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
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

digit_detection_model = torch.load(
    r'./saved_models/newest_digit_recognition_100.pkl')
digit_detection_model.to(device)
digit_detection_model.eval()


def no_overlap_pred(boxes, labels, scores, maxDim):
    minDistThresh = maxDim * 0.1
    # maxDistThresh = maxDim * 0.5
    # first get midpoint of all boxes
    midpoints = []
    for box in boxes:
        x_mid = (box[0] + box[2])/2
        y_mid = (box[1] + box[3])/2
        midpoints.append((x_mid, y_mid))
    # now select top three that are not close
    # we can always include top prediction
    indexesToUse = [0]
    for d in range(1, len(boxes)):
        # check distance between current index and other indexes in list
        usable = True
        for indexUsed in indexesToUse:
            # compute distance
            m1 = midpoints[indexUsed]
            m2 = midpoints[d]
            dist = math.sqrt(
                (math.pow((m1[0]-m2[0]), 2)+math.pow((m1[1]-m2[1]), 2)))
            # threshold dist
            if dist < minDistThresh:
                # too close or too far
                usable = False
        # if not skipped add to our indexes to use
        if usable and scores[d] >= 0.7:
            indexesToUse.append(d)
        # if we reached three can finish
        if len(indexesToUse) == 3:
            break

    # update lists
    boxes = boxes[indexesToUse]
    labels = labels[indexesToUse]
    scores = scores[indexesToUse]

    return boxes, labels, scores


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


def fig_num(img, number):
    # put number in bottom left corner of image
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv2.getTextSize(number, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)
    cv2.putText(img, number, (textX, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)


def fig_draw(img, box, label, score):
    # draw predicted bounding box and class label on the input image
    # draw predicted bounding box and class label on the input image
    xmin = round(box[0])
    ymin = round(box[1])
    xmax = round(box[2])
    ymax = round(box[3])

    predText = '' + str(label) + ':' + str(int(score*100)/100).lstrip('0')

    if label == 0:  # start with background as 0
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (181, 252, 131), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (181, 252, 131), thickness=2)
    elif label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), thickness=2)
    elif label == 2:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 255), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), thickness=2)
    elif label == 3:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 100, 255), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 100, 255), thickness=2)
    elif label == 4:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 142, 142), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 142, 142), thickness=2)
    elif label == 5:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 0, 255), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), thickness=2)
    elif label == 6:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 255), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), thickness=2)
    elif label == 7:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 255, 0), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), thickness=2)
    elif label == 8:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (121, 252, 206), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (121, 252, 206), thickness=2)
    elif label == 9:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (119, 118, 193), thickness=1)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (119, 118, 193), thickness=2)


def get_number_prediction(boxes, labels, scores, maxDim):
    # check if length is greater than three and if so select three most distinct
    if len(labels) >= 3:
        boxes, labels, scores = no_overlap_pred(boxes, labels, scores, maxDim)
    # if only 2 or less predictions, not usable
    if len(labels) < 3:
        return None, None, None, None
    # print('boxes:', boxes)

    # get indices to sort boxes by minimum x value
    sortInd = np.argsort(boxes[:, 0])
    # put labels in order
    labels = labels[sortInd]
    boxes = boxes[sortInd]
    scores = scores[sortInd]

    # create number
    number = ''
    # iterate through each label adding it to our number
    for i in range(len(labels)):
        label = str(labels[i].item())
        # check if 10 and covert to 0
        if label == '10':
            label = '0'
            labels[i] = 0
        number += label

    return number, boxes, labels, scores

def digit_recognition(img):
    annImg = img.copy()
    maxDim = max(annImg.shape[0], annImg.shape[1])
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    transform = T.Compose([T.ToTensor()])

    img = transform(img)

    img = np.array(normalize(img.numpy()), dtype=np.float32)
    img = torch.from_numpy(img)

    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = digit_detection_model([img.to(device)])

    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    # print(prediction)
    number, boxes, labels, scores = get_number_prediction(
        boxes, labels, scores, maxDim)

    if number is not None:
        # img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
        # img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
        # img = np.array(img)  # tensor -> ndarray

        # check if we have confusing digits
        # crop_confusing_digits(img, boxes, labels, scores)

        for i in range(len(boxes)):
            fig_draw(annImg, boxes[i], labels[i], scores[i])

        fig_num(annImg, number)

        return annImg, labels, scores

    return None, None, None

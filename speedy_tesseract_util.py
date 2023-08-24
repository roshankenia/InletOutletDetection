import cv2
import pytesseract
from pytesseract import Output
import os
import torch
import sys
import numpy as np
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

pytesseract.pytesseract.tesseract_cmd = r'../anaconda3/envs/tesseract2/bin/tesseract'
tessdata_dir_config = r'../anaconda3/envs/tesseract2/share/tessdata'
os.environ["TESSDATA_PREFIX"] = tessdata_dir_config


def preprocess(img):
    img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = 255-img  # invert image. tesseract prefers black text on white background

    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

    return img


def updateAccuracies(pebbleActualNumber, digitAccuracy, confusionMatrix, predLabels, predScores, img):
    print('labels:', predLabels, 'scores:', predScores)
    numberIsIncorrect = False
    scoreCode = ''
    for a in range(len(predLabels)):
        if predLabels[a].isdigit():
            actualDigit = pebbleActualNumber[a]
            predDigit = int(predLabels[a])
            predScore = predScores[a]

            # check if digit is correct
            if actualDigit == predDigit:
                # now update accordingly
                if predScore < 40:
                    digitAccuracy[1] += 1
                    scoreCode += '2'
                elif predScore >= 40 and predScore < 80:
                    digitAccuracy[3] += 1
                    scoreCode += '4'
                else:
                    digitAccuracy[5] += 1
                    scoreCode += '6'
                    confusionMatrix[actualDigit][predDigit] += 1
            else:
                numberIsIncorrect = True
                # now update accordingly
                if predScore < 40:
                    digitAccuracy[0] += 1
                    scoreCode += '1'
                elif predScore >= 40 and predScore < 80:
                    digitAccuracy[2] += 1
                    scoreCode += '3'
                else:
                    digitAccuracy[4] += 1
                    scoreCode += '5'
                    confusionMatrix[actualDigit][predDigit] += 1

    if numberIsIncorrect:
        digitAccuracy[6] += 1
        scoreCode += '7'
    else:
        digitAccuracy[7] += 1
        scoreCode += '8'

    # # put actual number in image
    # scoring = str(pebbleActualNumber[0]) + str(pebbleActualNumber[1]
    #                                            ) + str(pebbleActualNumber[2]) + ":" + scoreCode
    # # setup text
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # # get boundary of this text
    # textsize = cv2.getTextSize(scoring, font, 1, 2)[0]

    # # get coords based on boundary
    # textX = int((img.shape[1] - textsize[0]) / 2)
    # textY = int((img.shape[0] + textsize[1]) / 2)
    # cv2.putText(img, scoring, (textX, img.shape[0]-75), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (255, 255, 255), thickness=2)

    return digitAccuracy, confusionMatrix, img
# read through each image and predict


def tesseract_prediction(img):
    config = r'--oem 3 --psm 11 digits'
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(cv2.bitwise_not(img), (100, 100))
    # img = cv2.resize(img, (100, 100))
    img = preprocess(img)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
    n_boxes = len(d['level'])
    had_pred = False
    pred = None
    score = 0
    ind = -1
    # convert back to RGB and find best prediction
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(n_boxes):
        text = "".join(d["text"][i]).strip()
        conf = int(d["conf"][i])
        if conf > score:
            pred = text
            had_pred = True
            score = conf
            ind = i

    if not had_pred:
        cv2.putText(img, 'NONE', (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), thickness=10)
    else:
        (x, y, w, h) = (d['left'][ind], d['top']
                        [ind], d['width'][ind], d['height'][ind])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(img, str(text)+":"+str(score), (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 0), thickness=10)

    return img, pred, score


def tesseract_prediction_with_accuracy(img, pebbleActualNumber, digitAccuracy, confusionMatrix):
    config = r'--oem 3 --psm 11 digits'
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(cv2.bitwise_not(img), (100, 100))
    # img = cv2.resize(img, (100, 100))
    img = preprocess(img)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
    n_boxes = len(d['level'])
    had_pred = False
    pred = None
    score = 0
    ind = -1
    # convert back to RGB and find best prediction
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(n_boxes):
        text = "".join(d["text"][i]).strip()
        conf = int(d["conf"][i])
        if conf > score:
            pred = text
            had_pred = True
            score = conf
            ind = i

    if not had_pred or len(pred) != 3 or not pred.isdigit():
        font = cv2.FONT_HERSHEY_SIMPLEX
        # predText = str(text)+":"+str(score)
        predText = 'None'
        # get boundary of this text
        textsize = cv2.getTextSize(predText, font, 4, 5)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int((img.shape[0] + textsize[1]) / 2)
        cv2.putText(img, predText, (textX, img.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), thickness=5)
    else:
        # split into individual digits
        labels = [ch for ch in pred]
        scores = np.full(len(labels), score)
        (x, y, w, h) = (d['left'][ind], d['top']
                        [ind], d['width'][ind], d['height'][ind])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # predText = str(text)+":"+str(score)
        predText = str(text)
        # get boundary of this text
        textsize = cv2.getTextSize(predText, font, 4, 5)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int((img.shape[0] + textsize[1]) / 2)
        cv2.putText(img, predText, (textX, img.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (255, 255, 255), thickness=5)

        # add in scoring
        if len(labels) == 3:
            updateAccuracies(pebbleActualNumber, digitAccuracy, confusionMatrix,
                             labels, scores, img)

    return img, pred, score, digitAccuracy, confusionMatrix

    # [1, 1, 0, 4, 1, 4, 1, 3, 1, 0, 2, 2, 3, 3, 1, 7, 5, 11, 5, 7, 6, 2, 5, 3, 10, 3, 1, 1, 1, 2, 3, 4, 7, 5, 4, 3, 7, 6, 6, 5, 4, 8, 6, 5]

    # [2, 2, 1, 4, 5, 2, 3, 1, 2, 3, 3, 2, 4, 7, 8, 6, 10, 10, 15, 8, 12, 4, 4, 11, 7, 11, 11, 7, 5, 6, 8, 6, 5, 3, 6, 9, 7, 6, 14, 11, 8, 6, 12, 10, 14, 10, 2]

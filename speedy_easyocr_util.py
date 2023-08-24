import cv2
import easyocr
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


# create Jaided AI Easy OCR reader
reader = easyocr.Reader(['en'])


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
                if predScore < 0.5:
                    digitAccuracy[1] += 1
                    scoreCode += '2'
                elif predScore >= 0.5 and predScore < 0.75:
                    digitAccuracy[3] += 1
                    scoreCode += '4'
                else:
                    digitAccuracy[5] += 1
                    scoreCode += '6'
                    confusionMatrix[actualDigit][predDigit] += 1
            else:
                numberIsIncorrect = True
                # now update accordingly
                if predScore < 0.5:
                    digitAccuracy[0] += 1
                    scoreCode += '1'
                elif predScore >= 0.5 and predScore < 0.75:
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

    # put actual number in image
    scoring = str(pebbleActualNumber[0]) + str(pebbleActualNumber[1]
                                               ) + str(pebbleActualNumber[2]) + ":" + scoreCode
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv2.getTextSize(scoring, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)
    cv2.putText(img, scoring, (textX, img.shape[0]-75), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)

    return digitAccuracy, confusionMatrix, img
# read through each image and predict


def easy_prediction_with_accuracy(img, pebbleActualNumber, digitAccuracy, confusionMatrix):
    # predict on image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = preprocess(img)
    result = reader.readtext(img)
    print(result)
    # get best prediction
    n_boxes = len(result)
    had_pred = False
    pred = None
    score = 0
    ind = -1
    for i in range(n_boxes):
        text = "".join(result[i][1]).strip()
        conf = result[i][2]
        if conf > score:
            pred = text
            had_pred = True
            score = conf
            ind = i

    score = round(score, 4)
    print("PRED:::", pred)
    if pred is None:
        cv2.putText(img, 'NONE', (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), thickness=10)
    else:
        # split into individual digits
        labels = [ch for ch in pred]
        scores = np.full(len(labels), score)
        minx, miny = int(result[ind][0][0][0]), int(result[ind][0][0][1])
        maxx, maxy = int(result[ind][0][2][0]), int(result[ind][0][2][1])
        cv2.rectangle(img, (minx, miny), (maxx, maxy), (255, 0, 0), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        predText = str(text)+":"+str(score)
        # get boundary of this text
        textsize = cv2.getTextSize(predText, font, 1, 2)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int((img.shape[0] + textsize[1]) / 2)
        cv2.putText(img, predText, (textX, img.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), thickness=2)

        # add in scoring
        if len(labels) == 3:
            updateAccuracies(pebbleActualNumber, digitAccuracy, confusionMatrix,
                             labels, scores, img)

    return img, pred, score, digitAccuracy, confusionMatrix

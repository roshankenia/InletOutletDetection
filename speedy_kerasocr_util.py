import keras_ocr
import cv2
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


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = 255-img  # invert image. tesseract prefers black text on white background

    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def updateAccuracies(pebbleActualNumber, digitAccuracy, confusionMatrix, predLabels, img):
    print('labels:', predLabels)
    numberIsIncorrect = False
    scoreCode = ''
    for a in range(len(predLabels)):
        if predLabels[a].isdigit():
            actualDigit = pebbleActualNumber[a]
            predDigit = int(predLabels[a])

            # check if digit is correct
            if actualDigit == predDigit:
                # now update accordingly
                digitAccuracy[5] += 1
                scoreCode += '6'
                confusionMatrix[actualDigit][predDigit] += 1
            else:
                numberIsIncorrect = True
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


def patch_words(pred_groups):
    # need to sort pred groups by x value
    # split pred and coords
    preds = []
    for pred_group in pred_groups:
        preds.append([pred_group[1][0][0], pred_group[0]])
    # sort by first x value
    preds = sorted(preds)

    # create prediction only from numbers
    prediction = ''
    for (x, word) in preds:
        prediction += ''.join(filter(lambda i: i.isdigit(), word))
    return prediction

# read through each image and predict


def keras_prediction_with_accuracy(img, pebbleActualNumber, digitAccuracy, confusionMatrix):
    img = preprocess(img)

    prediction_groups = pipeline.recognize([img])
    prediction = None
    if len(prediction_groups) == 1:
        prediction = patch_words(prediction_groups[0])

    if prediction == None:
        cv2.putText(img, 'NONE', (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), thickness=10)
    else:
        # split into individual digits
        labels = [ch for ch in prediction]

        font = cv2.FONT_HERSHEY_SIMPLEX
        predText = "Pred:"+str(prediction)
        # get boundary of this text
        textsize = cv2.getTextSize(predText, font, 1, 2)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int((img.shape[0] + textsize[1]) / 2)
        cv2.putText(img, predText, (textX, img.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), thickness=2)

        # add in scoring
        if len(labels) == 3:
            updateAccuracies(pebbleActualNumber, digitAccuracy,
                             confusionMatrix, labels, img)

    return img, prediction, digitAccuracy, confusionMatrix

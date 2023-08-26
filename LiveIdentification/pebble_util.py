import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as VT
import torch
import math
# ensure we are running on the correct gpu
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('GPU not being used, exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class Pebble():
    def __init__(self, number, count, startTime):
        self.number = number
        self.firstSeen = count
        self.lastSeen = count
        self.startTime = startTime
        self.lastSeenTime = startTime
        self.digits = np.zeros((3, 10))
        self.currentDigitBoxes = None
        self.isConverged = False
        self.ConvergedClassification = '???'

        print('Pebble #'+str(self.number)+' has been created')

    def addDigits(self, labels, scores):
        for l in range(len(labels)):
            # only add if score is greater than 0.8
            if scores[l] >= 0.8:
                self.digits[l][labels[l]] += 1
            if scores[l] >= 0.98:
                self.digits[l][labels[l]] += 1
            # self.digits[l][labels[l]] += scores[l]
        # print('Pebble #'+str(self.number) +
        #       ' has updated digits '+str(self.digits))

    def addDigitBoxes(self, boxes, frameNumber, videoTime):
        self.currentDigitBoxes = boxes
        self.lastSeen = frameNumber
        self.lastSeenTime = videoTime

    def resetBoxes(self):
        self.currentDigitBoxes = None

    def obtainFinalClassification(self):
        if self.isConverged:
            return self.ConvergedClassification
        # check if no good prediction
        if np.sum(self.digits) == 0:
            return '???'
        # obtain top prediction for each digit position
        classification = ''
        converged = True
        for d in range(len(self.digits)):
            # take argmax of each position
            maxPos = np.argmax(self.digits[d])
            if self.digits[d][maxPos] == 0:
                # no maximum for this position
                return '???'
            elif self.digits[d][maxPos] < 10:
                converged = False
            classification += str(maxPos)

        # check if digits have converged
        if converged:
            self.isConverged = True
            self.ConvergedClassification = classification
        return classification
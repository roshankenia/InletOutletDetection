from speedy_tesseract_util import tesseract_prediction_with_accuracy
from speedy_tesseract_util import tesseract_prediction_with_accuracy_no_CLAHE
from speedy_pebble_util_tess import updatePebbleLocationTess
from speedy_easyocr_util import easy_prediction_with_accuracy
from speedy_easyocr_util import easy_prediction_with_accuracy_no_CLAHE
from speedy_pebble_util_easyocr import updatePebbleLocationEasyOCR
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
from speedy_detection_util_SVHN import showbox_with_accuracy
from speedy_crop_util import digit_segmentation
from speedy_pebble_util import updatePebbleLocationReg
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')
# c = 7000


def save_frame_and_mask(frame):
    folder = f"./FrameFolder/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    frame_folder = f"./FrameFolder/Frames/"
    if not os.path.isdir(frame_folder):
        os.mkdir(frame_folder)
    mask_folder = f"./FrameFolder/Masks/"
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)

    frame = cv2.resize(frame, (1920, 1080))
    global c
    # save img
    cv2.imwrite(frame_folder + "frame_" + str(c) + ".jpg", frame)

    # save mask
    # now create large background
    mask = np.zeros((1080, 1920, 3), np.uint8)
    cv2.imwrite(mask_folder + "frame_" + str(c) + ".jpg", mask)

    c += 1


class Video():
    def __init__(self, filename):
        self.activePebbles = []
        self.numOfPebbles = 0
        self.savedPebbles = []
        self.transform = T.Compose([T.PILToTensor()])

        self.vidcap = cv2.VideoCapture(
            f'./videos/Outlet Individual Pebble Videos/{filename}.MP4')
        self.filename = filename + '_FIGSIX_'
        self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        print(f'video {filename} has', str(
            self.frame_count), 'frames with an fps of', self.fps)
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('video dimensions width:', self.width, 'height:', self.height)

        self.imgFolder = f"./Individual Outlet Results/FigSixImages/"
        if not os.path.isdir(self.imgFolder):
            os.mkdir(self.imgFolder)

        # calculate distance threshold based on 25% of max video dimension
        self.distThreshold = int(0.5*max(self.width, self.height))
        print('distThresh is:', self.distThreshold)

    def print_final_classification(self):
        finClass = ''
        for pebble in self.activePebbles:
            print('Pebble classification:', pebble.obtainFinalClassification())
            finClass += pebble.obtainFinalClassification()
        return finClass

    def processNextFrame(self, frame, frameNumber, videoTime, pebbleActualNumber, digitAccuracy, confusionMatrix, inletSavedPebbles=None):
        og_frame = frame.copy()
        # check if image has digits with confidence
        pebbleDigitsCrops, pebbleDigitBoxes, pebbleDigitScores, goodPredictions, goodMasks, originalDigitCrops = digit_segmentation(
            frame)

        # see if digits were detected
        if pebbleDigitsCrops is not None:
            print('Frame with digits:', str(frameNumber))
            # update pebble location based on first pebble digit crop
            # tag and update pebble data
            currentPebble, self.activePebbles, self.numOfPebbles = updatePebbleLocationReg(
                pebbleDigitBoxes[0], self.activePebbles, self.distThreshold, self.numOfPebbles, frameNumber, videoTime)

            # update boxes
            currentPebble.addDigitBoxes(pebbleDigitBoxes)

            # check if converged already
            # if not currentPebble.isConverged:
            # save orientation bar prediction
            for i in range(len(pebbleDigitsCrops)):
                annImg, fixedImages = segment_and_fix_image_range(
                    pebbleDigitsCrops[i], originalDigitCrops[i], 0.9)
                # cv2.imwrite(os.path.join(self.imgFolder, "orgDigCrop_" +
                #             str(frameNumber) + "_num_"+str(i)+".jpg"), originalDigitCrops[i])
                for f in range(len(fixedImages)):
                    # downsize image
                    downsizedImage = fixedImages[f]
                    scale_percent = 30  # percent of original size
                    width = int(
                        downsizedImage.shape[1] * scale_percent / 100)
                    height = int(
                        downsizedImage.shape[0] * scale_percent / 100)
                    dim = (width, height)

                    downsizedImage = cv2.resize(
                        downsizedImage, dim, interpolation=cv2.INTER_AREA)
                    # prediciton
                    predImg, predlabels, predScores, digitAccuracy, confusionMatrix, indexReg = showbox_with_accuracy(
                        downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)

                    # Easy CLAHE prediciton
                    predImgEasyCLAHE, easyPredEasyCLAHE, easyScoreEasyCLAHE, digitAccuracy, confusionMatrix, indexEasyCLAHE = easy_prediction_with_accuracy(
                        downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)

                    predImgEasyNoCLAHE, easyPredEasyNoCLAHE, easyScoreEasyNoCLAHE, digitAccuracy, confusionMatrix, indexEasyNoCLAHE = easy_prediction_with_accuracy_no_CLAHE(
                        downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)

                    # Tess CLAHE prediciton
                    predImgTessCLAHE, tessPredTessCLAHE, tessScoreTessCLAHE, digitAccuracy, confusionMatrix, indexTessCLAHE = tesseract_prediction_with_accuracy(
                        downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)
                    predImgTessNoCLAHE, tessPredTessNoCLAHE, tessScoreTessNoCLAHE, digitAccuracy, confusionMatrix, indexTessNoCLAHE = tesseract_prediction_with_accuracy_no_CLAHE(
                        downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)
                    if indexReg == 2 and indexEasyCLAHE == 1 and indexEasyNoCLAHE == 0 and indexTessCLAHE == 1 and indexTessNoCLAHE == 0:
                        # save images
                        print('\n\nFOUND A SOLUTION\n\n')
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_REG.jpg"), predImg)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_EasyCLAHE.jpg"), predImgEasyCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_EasyNoCLAHE.jpg"), predImgEasyNoCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_TessCLAHE.jpg"), predImgTessCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_TessNoCLAHE.jpg"), predImgTessNoCLAHE)
                    elif indexReg == 2 and indexEasyCLAHE == 1 and indexEasyNoCLAHE == 0 and indexTessCLAHE == 0 and indexTessNoCLAHE == 0:
                        # save images
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_REG.jpg"), predImg)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_EasyCLAHE.jpg"), predImgEasyCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_EasyNoCLAHE.jpg"), predImgEasyNoCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_TessCLAHE.jpg"), predImgTessCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_TessNoCLAHE.jpg"), predImgTessNoCLAHE)
                    elif indexReg == 2 and indexEasyCLAHE == 0 and indexEasyNoCLAHE == 0 and indexTessCLAHE == 1 and indexTessNoCLAHE == 0:
                        # save images
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_REG.jpg"), predImg)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_EasyCLAHE.jpg"), predImgEasyCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_EasyNoCLAHE.jpg"), predImgEasyNoCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_TessCLAHE.jpg"), predImgTessCLAHE)
                        cv2.imwrite(os.path.join(self.imgFolder, "img_" + self.filename +
                                    str(frameNumber) + "_pred_"+str(f)+"_TessNoCLAHE.jpg"), predImgTessNoCLAHE)
                    print()


save_folder = f"./Individual Outlet Results/"
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

# obtain filenames from directory
videonames = list(
    sorted(os.listdir('./videos/Outlet Individual Pebble Videos/')))
accuracies = []
classifications = []
confusionMatrix = np.zeros((10, 10))
for videoname in videonames:
    # create inlet video
    videoname = videoname[:videoname.index('.')]
    pebbleNum = ''.join(filter(lambda i: i.isdigit(), videoname))
    print("VIDEO: ", pebbleNum)
    inletVideo = Video(pebbleNum)
    pebbleActualNumber = [int(dig) for dig in pebbleNum]
    digitAccuracy = np.zeros(8)
    # set frames count and fps
    num_frames = inletVideo.frame_count
    FPS = inletVideo.fps

    start = time.time()

    frameNumber = 0
    inletHasFrames, inletFrame = inletVideo.vidcap.read()
    while inletHasFrames:
        print('Processing frame #', frameNumber)
        videoTime = frameNumber/FPS
        # process inlet frame
        inletVideo.processNextFrame(
            inletFrame, frameNumber, videoTime, pebbleActualNumber, digitAccuracy, confusionMatrix)
        # check if we are currently processing
        # if none in frame can skip
        if len(inletVideo.activePebbles) == 0:
            # skip four frames
            for i in range(4):
                inletHasFrames, inletFrame = inletVideo.vidcap.read()
                frameNumber += 1
        inletHasFrames, inletFrame = inletVideo.vidcap.read()
        frameNumber += 1

    end = time.time()
    print('Total time elapsed:', (end-start))
    print('Digit Accuracy:', digitAccuracy)
    print("Videoname: ", videoname)
    print("Current Confusion Matrix:", confusionMatrix)
    finalClass = inletVideo.print_final_classification()
    classifications.append((pebbleNum, finalClass))
    accuracies.append(digitAccuracy)

    # When everything done, release the capture
    inletVideo.vidcap.release()
    print()
    print()

print('Final Results:')
print(classifications)
print(accuracies)
print(confusionMatrix)

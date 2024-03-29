from speedy_tesseract_util import tesseract_prediction_with_accuracy
from speedy_pebble_util_tess import updatePebbleLocationTess
from speedy_easyocr_util import easy_prediction_with_accuracy
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


# def save_frame_and_mask(frame):
#     folder = f"./FrameFolder/"
#     if not os.path.isdir(folder):
#         os.mkdir(folder)
#     frame_folder = f"./FrameFolder/Frames/"
#     if not os.path.isdir(frame_folder):
#         os.mkdir(frame_folder)
#     mask_folder = f"./FrameFolder/Masks/"
#     if not os.path.isdir(mask_folder):
#         os.mkdir(mask_folder)

#     frame = cv2.resize(frame, (1920, 1080))
#     global c
#     # save img
#     cv2.imwrite(frame_folder + "frame_" + str(c) + ".jpg", frame)

#     # save mask
#     # now create large background
#     mask = np.zeros((1080, 1920, 3), np.uint8)
#     cv2.imwrite(mask_folder + "frame_" + str(c) + ".jpg", mask)

#     c += 1


# class Video():
#     def __init__(self, filename):
#         self.activePebbles = []
#         self.numOfPebbles = 0
#         self.savedPebbles = []
#         self.transform = T.Compose([T.PILToTensor()])

#         self.vidcap = cv2.VideoCapture(
#             f'./videos/Outlet Individual Pebble Videos/{filename}.MP4')
#         filename = filename + '_SVHN_BASE'
#         self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
#         print(f'video {filename} has', str(
#             self.frame_count), 'frames with an fps of', self.fps)
#         self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print('video dimensions width:', self.width, 'height:', self.height)

#         folder = f"./Individual Outlet Results/{filename}/"
#         if not os.path.isdir(folder):
#             os.mkdir(folder)

#         # create demo video
#         self.processed_video = cv2.VideoWriter(f'./Individual Outlet Results/{filename}/processed_video.avi',
#                                                cv2.VideoWriter_fourcc(*'mp4v'), self.vidcap.get(cv2.CAP_PROP_FPS), (self.width, self.height))

#         self.imgFolder = f"./Individual Outlet Results/{filename}/Images/"
#         if not os.path.isdir(self.imgFolder):
#             os.mkdir(self.imgFolder)

#         # calculate distance threshold based on 25% of max video dimension
#         self.distThreshold = int(0.5*max(self.width, self.height))
#         print('distThresh is:', self.distThreshold)

#     def print_final_classification(self):
#         finClass = ''
#         for pebble in self.activePebbles:
#             print('Pebble classification:', pebble.obtainFinalClassification())
#             finClass += pebble.obtainFinalClassification()
#         return finClass

#     def processNextFrame(self, frame, frameNumber, videoTime, pebbleActualNumber, digitAccuracy, confusionMatrix, inletSavedPebbles=None):
#         og_frame = frame.copy()
#         # check if image has digits with confidence
#         pebbleDigitsCrops, pebbleDigitBoxes, pebbleDigitScores, goodPredictions, goodMasks, originalDigitCrops = digit_segmentation(
#             frame)

#         # see if digits were detected
#         if pebbleDigitsCrops is not None:
#             print('Frame with digits:', str(frameNumber))
#             # update pebble location based on first pebble digit crop
#             # tag and update pebble data
#             currentPebble, self.activePebbles, self.numOfPebbles = updatePebbleLocationReg(
#                 pebbleDigitBoxes[0], self.activePebbles, self.distThreshold, self.numOfPebbles, frameNumber, videoTime)

#             # update boxes
#             currentPebble.addDigitBoxes(pebbleDigitBoxes)

#             # check if converged already
#             # if not currentPebble.isConverged:
#             # save orientation bar prediction
#             for i in range(len(pebbleDigitsCrops)):
#                 annImg, fixedImages = segment_and_fix_image_range(
#                     pebbleDigitsCrops[i], originalDigitCrops[i], 0.9)
#                 # cv2.imwrite(os.path.join(self.imgFolder, "orgDigCrop_" +
#                 #             str(frameNumber) + "_num_"+str(i)+".jpg"), originalDigitCrops[i])
#                 for f in range(len(fixedImages)):
#                     # downsize image
#                     downsizedImage = fixedImages[f]
#                     scale_percent = 30  # percent of original size
#                     width = int(
#                         downsizedImage.shape[1] * scale_percent / 100)
#                     height = int(
#                         downsizedImage.shape[0] * scale_percent / 100)
#                     dim = (width, height)

#                     downsizedImage = cv2.resize(
#                         downsizedImage, dim, interpolation=cv2.INTER_AREA)
#                     # prediciton
#                     predImg, predlabels, predScores, digitAccuracy, confusionMatrix = showbox_with_accuracy(
#                         downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)
#                     if predImg is not None:
#                         cv2.imwrite(os.path.join(self.imgFolder, "img_" +
#                                     str(frameNumber) + "_pred_"+str(f)+".jpg"), predImg)
#                         # update digits
#                         currentPebble.addDigits(
#                             predlabels, predScores)
#         # create frame based on current active pebbles
#         if inletSavedPebbles is not None:
#             frameWithData = addToFrame(
#                 og_frame, self, frameNumber, videoTime, inletSavedPebbles)
#         else:
#             frameWithData = addToFrame(og_frame, self, frameNumber, videoTime)
#         # put frame into video
#         self.processed_video.write(frameWithData)


# def addToFrame(frame, video, frameNumber, videoTime, inletSavedPebbles=None):
#     height, width = frame.shape[:2]
#     if len(video.activePebbles) > 0:
#         # iterate through each active pebble and add their data in
#         for pebble in video.activePebbles:
#             # check if detected this frame
#             currentClassification = pebble.obtainFinalClassification()
#             if pebble.lastSeen == frameNumber and currentClassification != '???':
#                 # add in pebble detection area
#                 if pebble.currentPebbleBox is not None:
#                     minCord = (
#                         pebble.currentPebbleBox[0], pebble.currentPebbleBox[1])
#                     maxCord = (
#                         pebble.currentPebbleBox[2], pebble.currentPebbleBox[3])
#                     cv2.rectangle(frame, minCord, maxCord,
#                                   color=(0, 255, 0), thickness=4)
#                     # put pebble number
#                     cv2.putText(frame, 'Pebble #'+str(pebble.number), minCord, cv2.FONT_HERSHEY_SIMPLEX,
#                                 2, (0, 255, 0), thickness=2)

#                     # put highest predicted digits in center
#                     bottomCenterCord = (
#                         int(((minCord[0]+maxCord[0])/2)-200), int(maxCord[1]))

#                     cv2.putText(frame, 'Pred: '+str(currentClassification),
#                                 bottomCenterCord, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), thickness=3)
#                 # add in digit detection area
#                 if pebble.currentDigitBoxes is not None:
#                     for digitBox in pebble.currentDigitBoxes:
#                         minCord = (digitBox[0], digitBox[1])
#                         maxCord = (digitBox[2], digitBox[3])
#                         cv2.rectangle(frame, minCord, maxCord,
#                                       color=(0, 255, 255), thickness=3)
#                 # reset current boxes
#                 pebble.resetBoxes()
#     if inletSavedPebbles is not None:
#         # add in info about inlet saved pebbles
#         cv2.putText(frame, 'Inlet Pebbles:', (750, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), thickness=4)
#         lineNum = 0
#         for i in range(len(inletSavedPebbles)):
#             text = ''+inletSavedPebbles[i][0]+': '+inletSavedPebbles[i][1]
#             place = None
#             if i % 2 == 0:
#                 place = (750, 85+35*lineNum)
#             else:
#                 place = (1050, 85+35*lineNum)
#                 lineNum += 1
#             cv2.putText(frame, text, place,
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=4)
#     # add in info about saved pebbles
#     cv2.putText(frame, 'Pebble Last Seen:', (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=4)
#     lineNum = 0
#     for i in range(len(video.savedPebbles)):
#         text = ''+video.savedPebbles[i][0]+': '+video.savedPebbles[i][1]
#         place = None
#         if i % 2 == 0:
#             place = (50, 85+35*lineNum)
#         else:
#             place = (350, 85+35*lineNum)
#             lineNum += 1
#         cv2.putText(frame, text, place,
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=4)

#     # add in time
#     cv2.putText(frame, str(round(videoTime, 2))+'s', (width-200, height-75), cv2.FONT_HERSHEY_SIMPLEX,
#                 2, (255, 255, 255), thickness=3)
#     return frame


# save_folder = f"./Individual Outlet Results/"
# if not os.path.isdir(save_folder):
#     os.mkdir(save_folder)

# # obtain filenames from directory
# videonames = list(
#     sorted(os.listdir('./videos/Outlet Individual Pebble Videos/')))
# accuracies = []
# classifications = []
# confusionMatrix = np.zeros((10, 10))
# for videoname in videonames:
#     # create inlet video
#     videoname = videoname[:videoname.index('.')]
#     pebbleNum = ''.join(filter(lambda i: i.isdigit(), videoname))
#     print("VIDEO: ", pebbleNum)
#     inletVideo = Video(pebbleNum)
#     pebbleActualNumber = [int(dig) for dig in pebbleNum]
#     digitAccuracy = np.zeros(8)
#     # set frames count and fps
#     num_frames = inletVideo.frame_count
#     FPS = inletVideo.fps

#     start = time.time()

#     frameNumber = 0
#     inletHasFrames, inletFrame = inletVideo.vidcap.read()
#     while inletHasFrames:
#         print('Processing frame #', frameNumber)
#         videoTime = frameNumber/FPS
#         # process inlet frame
#         inletVideo.processNextFrame(
#             inletFrame, frameNumber, videoTime, pebbleActualNumber, digitAccuracy, confusionMatrix)
#         # check if we are currently processing
#         # if none in frame can skip
#         if len(inletVideo.activePebbles) == 0:
#             # skip four frames
#             for i in range(4):
#                 inletHasFrames, inletFrame = inletVideo.vidcap.read()
#                 frameNumber += 1
#         # else:
#         #     # if all converged can skip
#         #     convergedNum = 0
#         #     for actPebble in inletVideo.activePebbles:
#         #         if actPebble.isConverged:
#         #             convergedNum += 1
#         #     if convergedNum == len(inletVideo.activePebbles):
#         #         # skip four frames
#         #         for i in range(4):
#         #             inletHasFrames, inletFrame = inletVideo.vidcap.read()
#         #             frameNumber += 1
#         inletHasFrames, inletFrame = inletVideo.vidcap.read()
#         frameNumber += 1

#     end = time.time()
#     print('Total time elapsed:', (end-start))
#     print('Digit Accuracy:', digitAccuracy)
#     print("Videoname: ", videoname)
#     print("Current Confusion Matrix:", confusionMatrix)
#     finalClass = inletVideo.print_final_classification()
#     classifications.append((pebbleNum, finalClass))
#     accuracies.append(digitAccuracy)

#     # When everything done, release the capture
#     inletVideo.vidcap.release()
#     inletVideo.processed_video.release()

#     print()
#     print()

# print('Final Results:')
# print(classifications)
# print(accuracies)
# print(confusionMatrix)


# # ensure we are running on the correct gpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
# if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
#     print('exiting')
#     sys.exit()
# else:
#     print('GPU is being properly used')
# c = 7000


# class Video():
#     def __init__(self, filename):
#         self.activePebbles = []
#         self.numOfPebbles = 0
#         self.savedPebbles = []
#         self.transform = T.Compose([T.PILToTensor()])

#         self.vidcap = cv2.VideoCapture(
#             f'./videos/Outlet Individual Pebble Videos/{filename}.MP4')
#         filename = filename+'_EasyOCRCLAHE'
#         self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
#         print(f'video {filename} has', str(
#             self.frame_count), 'frames with an fps of', self.fps)
#         self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print('video dimensions width:', self.width, 'height:', self.height)

#         folder = f"./Individual Outlet Results/{filename}/"
#         if not os.path.isdir(folder):
#             os.mkdir(folder)

#         # create demo video
#         self.processed_video = cv2.VideoWriter(f'./Individual Outlet Results/{filename}/processed_video.avi',
#                                                cv2.VideoWriter_fourcc(*'mp4v'), self.vidcap.get(cv2.CAP_PROP_FPS), (self.width, self.height))

#         self.imgFolder = f"./Individual Outlet Results/{filename}/Images/"
#         if not os.path.isdir(self.imgFolder):
#             os.mkdir(self.imgFolder)

#         # calculate distance threshold based on 25% of max video dimension
#         self.distThreshold = int(0.5*max(self.width, self.height))
#         print('distThresh is:', self.distThreshold)

#     def print_final_classification(self):
#         finClass = ''
#         for pebble in self.activePebbles:
#             print('Pebble classification:', pebble.obtainFinalClassification())
#             finClass += pebble.obtainFinalClassification()
#         return finClass

#     def processNextFrame(self, frame, frameNumber, videoTime, pebbleActualNumber, digitAccuracy, confusionMatrix, inletSavedPebbles=None):
#         og_frame = frame.copy()
#         # check if image has digits with confidence
#         pebbleDigitsCrops, pebbleDigitBoxes, pebbleDigitScores, goodPredictions, goodMasks, originalDigitCrops = digit_segmentation(
#             frame)

#         # see if digits were detected
#         if pebbleDigitsCrops is not None:
#             print('Frame with digits:', str(frameNumber))
#             # update pebble location based on first pebble digit crop
#             # tag and update pebble data
#             currentPebble, self.activePebbles, self.numOfPebbles = updatePebbleLocationEasyOCR(
#                 pebbleDigitBoxes[0], self.activePebbles, self.distThreshold, self.numOfPebbles, frameNumber, videoTime)

#             # update boxes
#             currentPebble.addDigitBoxes(pebbleDigitBoxes)

#             # check if converged already
#             # if not currentPebble.isConverged:
#             # save orientation bar prediction
#             for i in range(len(pebbleDigitsCrops)):
#                 annImg, fixedImages = segment_and_fix_image_range(
#                     pebbleDigitsCrops[i], originalDigitCrops[i], 0.9)
#                 for f in range(len(fixedImages)):
#                     # downsize image
#                     downsizedImage = fixedImages[f]
#                     scale_percent = 30  # percent of original size
#                     width = int(
#                         downsizedImage.shape[1] * scale_percent / 100)
#                     height = int(
#                         downsizedImage.shape[0] * scale_percent / 100)
#                     dim = (width, height)

#                     downsizedImage = cv2.resize(
#                         downsizedImage, dim, interpolation=cv2.INTER_AREA)
#                     # prediciton
#                     predImg, easyPred, easyScore, digitAccuracy, confusionMatrix = easy_prediction_with_accuracy(
#                         downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)
#                     if easyPred is not None:
#                         # update digits
#                         currentPebble.addDigits(easyPred, easyScore)
#                         cv2.imwrite(os.path.join(self.imgFolder, "img_" +
#                                     str(frameNumber) + "_pred_"+str(f)+".jpg"), predImg)

#         # create frame based on current active pebbles
#         if inletSavedPebbles is not None:
#             frameWithData = addToFrame(
#                 og_frame, self, frameNumber, videoTime, inletSavedPebbles)
#         else:
#             frameWithData = addToFrame(og_frame, self, frameNumber, videoTime)
#         # put frame into video
#         self.processed_video.write(frameWithData)


# def addToFrame(frame, video, frameNumber, videoTime, inletSavedPebbles=None):
#     height, width = frame.shape[:2]
#     if len(video.activePebbles) > 0:
#         # iterate through each active pebble and add their data in
#         for pebble in video.activePebbles:
#             # check if detected this frame
#             currentClassification = pebble.obtainFinalClassification()
#             if pebble.lastSeen == frameNumber and currentClassification != '???':
#                 # add in pebble detection area
#                 if pebble.currentPebbleBox is not None:
#                     minCord = (
#                         pebble.currentPebbleBox[0], pebble.currentPebbleBox[1])
#                     maxCord = (
#                         pebble.currentPebbleBox[2], pebble.currentPebbleBox[3])
#                     cv2.rectangle(frame, minCord, maxCord,
#                                   color=(0, 255, 0), thickness=4)
#                     # put pebble number
#                     cv2.putText(frame, 'Pebble #'+str(pebble.number), minCord, cv2.FONT_HERSHEY_SIMPLEX,
#                                 2, (0, 255, 0), thickness=2)

#                     # put highest predicted digits in center
#                     bottomCenterCord = (
#                         int(((minCord[0]+maxCord[0])/2)-200), int(maxCord[1]))

#                     cv2.putText(frame, 'Pred: '+str(currentClassification),
#                                 bottomCenterCord, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), thickness=3)
#                 # add in digit detection area
#                 if pebble.currentDigitBoxes is not None:
#                     for digitBox in pebble.currentDigitBoxes:
#                         minCord = (digitBox[0], digitBox[1])
#                         maxCord = (digitBox[2], digitBox[3])
#                         cv2.rectangle(frame, minCord, maxCord,
#                                       color=(0, 255, 255), thickness=3)
#                 # reset current boxes
#                 pebble.resetBoxes()
#     if inletSavedPebbles is not None:
#         # add in info about inlet saved pebbles
#         cv2.putText(frame, 'Inlet Pebbles:', (750, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), thickness=4)
#         lineNum = 0
#         for i in range(len(inletSavedPebbles)):
#             text = ''+inletSavedPebbles[i][0]+': '+inletSavedPebbles[i][1]
#             place = None
#             if i % 2 == 0:
#                 place = (750, 85+35*lineNum)
#             else:
#                 place = (1050, 85+35*lineNum)
#                 lineNum += 1
#             cv2.putText(frame, text, place,
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=4)
#     # add in info about saved pebbles
#     cv2.putText(frame, 'Pebble Last Seen:', (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=4)
#     lineNum = 0
#     for i in range(len(video.savedPebbles)):
#         text = ''+video.savedPebbles[i][0]+': '+video.savedPebbles[i][1]
#         place = None
#         if i % 2 == 0:
#             place = (50, 85+35*lineNum)
#         else:
#             place = (350, 85+35*lineNum)
#             lineNum += 1
#         cv2.putText(frame, text, place,
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=4)

#     # add in time
#     cv2.putText(frame, str(round(videoTime, 2))+'s', (width-200, height-75), cv2.FONT_HERSHEY_SIMPLEX,
#                 2, (255, 255, 255), thickness=3)
#     return frame


# save_folder = f"./Individual Outlet Results/"
# if not os.path.isdir(save_folder):
#     os.mkdir(save_folder)

# # obtain filenames from directory
# videonames = list(
#     sorted(os.listdir('./videos/Outlet Individual Pebble Videos/')))
# accuracies = []
# classifications = []
# confusionMatrix = np.zeros((10, 10))
# for videoname in videonames:
#     # create inlet video
#     videoname = videoname[:videoname.index('.')]
#     pebbleNum = ''.join(filter(lambda i: i.isdigit(), videoname))
#     print("VIDEO: ", pebbleNum)
#     inletVideo = Video(pebbleNum)
#     pebbleActualNumber = [int(dig) for dig in pebbleNum]
#     digitAccuracy = np.zeros(8)
#     # set frames count and fps
#     num_frames = inletVideo.frame_count
#     FPS = inletVideo.fps

#     start = time.time()

#     frameNumber = 0
#     inletHasFrames, inletFrame = inletVideo.vidcap.read()
#     while inletHasFrames:
#         print('Processing frame #', frameNumber)
#         videoTime = frameNumber/FPS
#         # process inlet frame
#         inletVideo.processNextFrame(
#             inletFrame, frameNumber, videoTime, pebbleActualNumber, digitAccuracy, confusionMatrix)
#         # check if we are currently processing
#         # if none in frame can skip
#         if len(inletVideo.activePebbles) == 0:
#             # skip four frames
#             for i in range(4):
#                 inletHasFrames, inletFrame = inletVideo.vidcap.read()
#                 frameNumber += 1
#         # else:
#         #     # if all converged can skip
#         #     convergedNum = 0
#         #     for actPebble in inletVideo.activePebbles:
#         #         if actPebble.isConverged:
#         #             convergedNum += 1
#         #     if convergedNum == len(inletVideo.activePebbles):
#         #         # skip four frames
#         #         for i in range(4):
#         #             inletHasFrames, inletFrame = inletVideo.vidcap.read()
#         #             frameNumber += 1
#         inletHasFrames, inletFrame = inletVideo.vidcap.read()
#         frameNumber += 1

#     end = time.time()
#     print('Total time elapsed:', (end-start))
#     print('Digit Accuracy:', digitAccuracy)
#     print("Videoname: ", videoname)
#     print("Current Confusion Matrix:", confusionMatrix)
#     finalClass = inletVideo.print_final_classification()
#     classifications.append((pebbleNum, finalClass))
#     accuracies.append(digitAccuracy)

#     # When everything done, release the capture
#     inletVideo.vidcap.release()
#     inletVideo.processed_video.release()

#     print()
#     print()

# print('Final Results:')
# print(classifications)
# print(accuracies)
# print(confusionMatrix)


# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')
c = 7000


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
        filename = filename+'_TessNoNeCLAHE'
        self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        print(f'video {filename} has', str(
            self.frame_count), 'frames with an fps of', self.fps)
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('video dimensions width:', self.width, 'height:', self.height)

        folder = f"./Individual Outlet Results/{filename}/"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # create demo video
        self.processed_video = cv2.VideoWriter(f'./Individual Outlet Results/{filename}/processed_video.avi',
                                               cv2.VideoWriter_fourcc(*'mp4v'), self.vidcap.get(cv2.CAP_PROP_FPS), (self.width, self.height))

        self.imgFolder = f"./Individual Outlet Results/{filename}/Images/"
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
            currentPebble, self.activePebbles, self.numOfPebbles = updatePebbleLocationTess(
                pebbleDigitBoxes[0], self.activePebbles, self.distThreshold, self.numOfPebbles, frameNumber, videoTime)

            # update boxes
            currentPebble.addDigitBoxes(pebbleDigitBoxes)

            # check if converged already
            # if not currentPebble.isConverged:
            # save orientation bar prediction
            for i in range(len(pebbleDigitsCrops)):
                annImg, fixedImages = segment_and_fix_image_range(
                    pebbleDigitsCrops[i], originalDigitCrops[i], 0.9)
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
                    predImg, tessPred, tessScore, digitAccuracy, confusionMatrix = tesseract_prediction_with_accuracy(
                        downsizedImage, pebbleActualNumber, digitAccuracy, confusionMatrix)
                    # if tessPred is not None:
                    # update digits
                    print('updating')
                    # currentPebble.addDigits(tessPred, tessScore)
                    cv2.imwrite(os.path.join(self.imgFolder, "img_" +
                                str(frameNumber) + "_pred_"+str(f)+".jpg"), predImg)

        # create frame based on current active pebbles
        if inletSavedPebbles is not None:
            frameWithData = addToFrame(
                og_frame, self, frameNumber, videoTime, inletSavedPebbles)
        else:
            frameWithData = addToFrame(og_frame, self, frameNumber, videoTime)
        # put frame into video
        self.processed_video.write(frameWithData)


def addToFrame(frame, video, frameNumber, videoTime, inletSavedPebbles=None):
    height, width = frame.shape[:2]
    if len(video.activePebbles) > 0:
        # iterate through each active pebble and add their data in
        for pebble in video.activePebbles:
            # check if detected this frame
            currentClassification = pebble.obtainFinalClassification()
            if pebble.lastSeen == frameNumber and currentClassification != '???':
                # add in pebble detection area
                if pebble.currentPebbleBox is not None:
                    minCord = (
                        pebble.currentPebbleBox[0], pebble.currentPebbleBox[1])
                    maxCord = (
                        pebble.currentPebbleBox[2], pebble.currentPebbleBox[3])
                    cv2.rectangle(frame, minCord, maxCord,
                                  color=(0, 255, 0), thickness=4)
                    # put pebble number
                    cv2.putText(frame, 'Pebble #'+str(pebble.number), minCord, cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 0), thickness=2)

                    # put highest predicted digits in center
                    bottomCenterCord = (
                        int(((minCord[0]+maxCord[0])/2)-200), int(maxCord[1]))

                    cv2.putText(frame, 'Pred: '+str(currentClassification),
                                bottomCenterCord, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), thickness=3)
                # add in digit detection area
                if pebble.currentDigitBoxes is not None:
                    for digitBox in pebble.currentDigitBoxes:
                        minCord = (digitBox[0], digitBox[1])
                        maxCord = (digitBox[2], digitBox[3])
                        cv2.rectangle(frame, minCord, maxCord,
                                      color=(0, 255, 255), thickness=3)
                # reset current boxes
                pebble.resetBoxes()
    if inletSavedPebbles is not None:
        # add in info about inlet saved pebbles
        cv2.putText(frame, 'Inlet Pebbles:', (750, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), thickness=4)
        lineNum = 0
        for i in range(len(inletSavedPebbles)):
            text = ''+inletSavedPebbles[i][0]+': '+inletSavedPebbles[i][1]
            place = None
            if i % 2 == 0:
                place = (750, 85+35*lineNum)
            else:
                place = (1050, 85+35*lineNum)
                lineNum += 1
            cv2.putText(frame, text, place,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=4)
    # add in info about saved pebbles
    cv2.putText(frame, 'Pebble Last Seen:', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=4)
    lineNum = 0
    for i in range(len(video.savedPebbles)):
        text = ''+video.savedPebbles[i][0]+': '+video.savedPebbles[i][1]
        place = None
        if i % 2 == 0:
            place = (50, 85+35*lineNum)
        else:
            place = (350, 85+35*lineNum)
            lineNum += 1
        cv2.putText(frame, text, place,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=4)

    # add in time
    cv2.putText(frame, str(round(videoTime, 2))+'s', (width-200, height-75), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), thickness=3)
    return frame


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
    if videoname == '346.MP4':
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
            # else:
            #     # if all converged can skip
            #     convergedNum = 0
            #     for actPebble in inletVideo.activePebbles:
            #         if actPebble.isConverged:
            #             convergedNum += 1
            #     if convergedNum == len(inletVideo.activePebbles):
            #         # skip four frames
            #         for i in range(4):
            #             inletHasFrames, inletFrame = inletVideo.vidcap.read()
            #             frameNumber += 1
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
        inletVideo.processed_video.release()

        print()
        print()

print('Final Results:')
print(classifications)
print(accuracies)
print(confusionMatrix)

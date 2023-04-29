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

from pebble_segmentation_util import pebble_segmentation, create_full_frame_crop
from pebble_util import updatePebbleLocation
from digit_segmentation_util import digit_segmentation
from crop_orientation_util import find_usable_crops
from digit_detection_util import individual_digit_detection
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class Video():
    def __init__(self, filename):
        self.activePebbles = []
        self.numOfPebbles = 0
        self.savedPebbles = []
        self.transform = T.Compose([T.PILToTensor()])

        self.vidcap = cv2.VideoCapture(f'./videos/{filename}.MP4')
        self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        print(f'video {filename} has', str(
            self.frame_count), 'frames with an fps of', self.fps)
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('video dimensions width:', self.width, 'height:', self.height)

        folder = f"./io_results/{filename}/"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # create demo video
        self.processed_video = cv2.VideoWriter(f'./io_results/{filename}/processed_video.avi',
                                               cv2.VideoWriter_fourcc(*'mp4v'), self.vidcap.get(cv2.CAP_PROP_FPS), (self.width, self.height))

        self.imgFolder = f"./io_results/{filename}/Images/"
        if not os.path.isdir(self.imgFolder):
            os.mkdir(self.imgFolder)

        # calculate distance threshold based on 25% of max video dimension
        self.distThreshold = int(0.25*max(self.width, self.height))
        print('distThresh is:', self.distThreshold)

    def removeInactive(self, frameNumber):
        # remove all inactive pebbles
        pebblesToKeep = []
        for pebble in self.activePebbles:
            if frameNumber - pebble.lastSeen <= 3:
                pebblesToKeep.append(pebble)
            else:
                # save pebble to match between inlet and outlet
                savePebble = (pebble.obtainFinalClassification(),
                              str(round(pebble.lastSeenTime, 3)))
                self.savedPebbles.append(savePebble)

        # set active pebbles
        self.activePebbles = pebblesToKeep

    def processNextFrame(self, frame, frameNumber, videoTime, inletSavedPebbles=None):
        og_frame = frame.copy()
        # check if image has a pebble with confidence
        masks, boxes, pred_cls = pebble_segmentation(frame)
        if masks is not None:
            # pebble detected, use most confident mask
            pebbleMask = masks[0]
            pebbleBox = [boxes[0][0][0], boxes[0][0]
                         [1], boxes[0][1][0], boxes[0][1][1]]
            pebbelPredClass = pred_cls[0]

            # tag and update pebble data
            currentPebble, self.activePebbles, self.numOfPebbles = updatePebbleLocation(
                pebbleBox, self.activePebbles, self.distThreshold, self.numOfPebbles, frameNumber, videoTime)

            # update pebble box
            currentPebble.addPebbleBox(pebbleBox)

            # focus on pebble area in video
            pebbleDetectionCrop = create_full_frame_crop(frame, pebbleMask)

            # create into PIL image
            pebbleDetectionCrop = Image.fromarray(pebbleDetectionCrop)
            pebbleDetectionCrop, _ = self.transform(pebbleDetectionCrop, None)

            # now try to obtain digit crop
            pebbleDigitsCrops, pebbleDigitBoxes = digit_segmentation(
                pebbleDetectionCrop)

            # see if digits were detected
            if pebbleDigitsCrops is not None:
                # add first box
                currentPebble.addDigitBoxes(pebbleDigitBoxes)
                # rotate crops and only save usable ones
                usablePebbleDigitsCrops = find_usable_crops(
                    pebbleDigitsCrops, frameNumber, self.imgFolder)

                # now we try to predict on the usable digit crops
                individual_digit_detection(
                    usablePebbleDigitsCrops, self.imgFolder, self.transform, currentPebble)
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
            if pebble.lastSeen == frameNumber:
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
                    predText = 'Pred: '+str(pebble.obtainFinalClassification())

                    # get boundary of this text
                    textsize = cv2.getTextSize(
                        predText, cv2.FONT_HERSHEY_SIMPLEX, 4, 3)[0]

                    bottomCenterCord = (
                        int(((minCord[0]+maxCord[0])/2)-20), int(maxCord[1]))

                    # get coords based on boundary
                    textX = int((bottomCenterCord[1] - textsize[0]) / 2)
                    textY = int((bottomCenterCord[0] + textsize[1]) / 2)

                    cv2.putText(frame, predText,
                                (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), thickness=3)
                # add in digit detection area
                if pebble.currentDigitBoxes is not None:
                    for digitBox in pebble.currentDigitBoxes:
                        minCord = (digitBox[0], digitBox[1])
                        maxCord = (digitBox[2], digitBox[3])
                        cv2.rectangle(frame, minCord, maxCord,
                                      color=(0, 255, 255), thickness=3)
                        # put predicted class
                        cv2.putText(
                            frame, 'digits', minCord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=2)
                # reset current boxes
                pebble.resetBoxes()
    if inletSavedPebbles is not None:
        # add in info about inlet saved pebbles
        cv2.putText(frame, 'Inlet Pebbles:', (750, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), thickness=4)
        for i in range(len(inletSavedPebbles)):
            text = ''+inletSavedPebbles[i][0]+': '+inletSavedPebbles[i][1]
            placeIndex = i % 4
            place = None
            if placeIndex == 0:
                place = (750, 50+35*(i+2))
            elif placeIndex == 1:
                place = (950, 50+35*(i+1))
            elif placeIndex == 2:
                place = (750, 50+35*(i+2))
            elif placeIndex == 3:
                place = (950, 50+35*(i+1))

            cv2.putText(frame, text, place,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=4)
    # add in info about saved pebbles
    cv2.putText(frame, 'Pebble Last Seen:', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=4)
    for i in range(len(video.savedPebbles)):
        text = ''+video.savedPebbles[i][0]+': '+video.savedPebbles[i][1]
        placeIndex = i % 4
        place = None
        if placeIndex == 0:
            place = (750, 50+35*(i+2))
        elif placeIndex == 1:
            place = (950, 50+35*(i+1))
        elif placeIndex == 2:
            place = (750, 50+35*(i+2))
        elif placeIndex == 3:
            place = (950, 50+35*(i+1))
        cv2.putText(frame, text, place,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=4)

    # add in time
    cv2.putText(frame, str(round(videoTime, 2))+'s', (width-200, height-75), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), thickness=3)
    return frame


# create inlet video
inletVideo = Video('S1060001In')
# create outlet video
outletVideo = Video('S1060001Out')
# ensure videos have same frame count and FPS
if inletVideo.frame_count != outletVideo.frame_count or inletVideo.fps != outletVideo.fps:
    sys.exit('Videos are not in sync.')

# set frames count and fps
num_frames = inletVideo.frame_count
FPS = inletVideo.fps

start = time.time()


for frameNumber in range(num_frames):
    print('Processing frame #', frameNumber)
    videoTime = frameNumber/FPS
    inletHasFrames, inletFrame = inletVideo.vidcap.read()
    outletHasFrames, outletFrame = outletVideo.vidcap.read()
    if inletHasFrames and outletHasFrames:
        # process inlet frame
        inletVideo.processNextFrame(inletFrame, frameNumber, videoTime)

        # process outlet frame
        outletVideo.processNextFrame(
            outletFrame, frameNumber, videoTime, inletVideo.savedPebbles)
    elif inletHasFrames or outletHasFrames:
        sys.exit('Videos are not in sync.')
    else:
        break

    inletVideo.removeInactive(frameNumber)
    outletVideo.removeInactive(frameNumber)

end = time.time()
print('Total time elapsed:', (end-start))

# When everything done, release the capture
inletVideo.vidcap.release()
outletVideo.vidcap.release()
inletVideo.processed_video.release()
outletVideo.processed_video.release()
cv2.destroyAllWindows()

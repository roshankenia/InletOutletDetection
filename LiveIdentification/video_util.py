import sys
import os
import cv2
import numpy as np
import torch
import sys_path
import training_utilities.transforms as T

from DAOA_util import digit_area_orientation_alignment
from DR_util import digit_recognition
from DAD_util import digit_area_detection
from pebble_util import Pebble
# ensure we are running on the correct gpu
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('GPU not being used, exiting')
    sys.exit()
else:
    print('GPU is being properly used')
c = 7000


# constant variables for timings to subtract from residence time to get exact time
avgTimeAugerToColumnInlet = np.mean([3.28, 3.29, 3.16])
print('avgTimeAugerToColumnInlet:', avgTimeAugerToColumnInlet)
avgTimeColumnExitToAuger = np.mean([10.57, 10.95, 11.08])
print('avgTimeColumnExitToAuger:', avgTimeColumnExitToAuger)



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


def square_and_resize(img, resize):
    ch, cw = img.shape[:2]
    imgSize = max(ch, cw)

    # create background
    background = np.zeros((imgSize, imgSize, 3), np.uint8)

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((imgSize-ch)/2)
    xoff = round((imgSize-cw)/2)

    # place crop in image
    background[yoff:yoff+ch, xoff:xoff+cw] = img

    # resize digitCrop
    background = cv2.resize(background, (resize, resize))

    return background


class Video():
    def __init__(self, videoname, save_folder, isInlet):
        self.numOfPebbles = 0
        self.activePebble = None
        self.pebbleLastSeen = 0
        self.isInlet = isInlet
        self.transform = T.Compose([T.PILToTensor()])

        vid_path = os.path.join("./Carved After Videos/", videoname+".MP4")
        self.vidcap = cv2.VideoCapture(vid_path)
        self.frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        print(f'video {videoname} has', str(
            self.frame_count), 'frames with an fps of', self.fps)
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('video dimensions width:', self.width, 'height:', self.height)

        # create demo video
        dem_path = os.path.join(save_folder, "demo_video.avi")
        self.processed_video = cv2.VideoWriter(dem_path, cv2.VideoWriter_fourcc(
            *'mp4v'), self.vidcap.get(cv2.CAP_PROP_FPS), (self.width, self.height))

        self.imgFolder = os.path.join(save_folder, "Images")
        if not os.path.isdir(self.imgFolder):
            os.mkdir(self.imgFolder)

    def processPebble(self, frameNumber, inletPebbles, outletUnrecognizedPebbles, identifiedPebbles, lock):
        # update if pebble is there or not if last seen more than 2 seconds ago
        if self.activePebble is not None and frameNumber - self.pebbleLastSeen > self.fps * 2:
            # lock pebble arrays
            lock.acquire()

            # get pebble final identification
            finalIdent = self.activePebble.obtainFinalClassification()
            finalTime = self.activePebble.lastSeenTime
            # add to inletPebbles if inlet stream
            if self.isInlet:
                inletPebbles.append((finalIdent, finalTime))
            else:
                # if outlet look for matching pebble in inletPebbles
                found = -1
                for i in range(len(inletPebbles)):
                    if inletPebbles[i][0] == finalIdent:
                        # match found
                        found = i
                        break

                # check if found and process residence time
                if found != -1:
                    outletTime = finalTime
                    inletTime = inletPebbles[found][1]

                    residenceTime = outletTime - inletTime - \
                        avgTimeAugerToColumnInlet - avgTimeColumnExitToAuger

                    print('Pebble', finalIdent,
                          'had a residence time of:', residenceTime)
                    identifiedPebbles.append(
                        (finalIdent, inletTime, outletTime))

                    # remove it from inletPebbles
                    del inletPebbles[found]
                else:
                    # if not found, need to add to outletUnrecognizedPebbles
                    outletUnrecognizedPebbles.append((finalIdent, finalTime))

            # unlock pebble arrays
            lock.release()

            self.activePebble = None

    def processNextFrame(self, frame, frameNumber, videoTime):
        og_frame = frame.copy()
        # check if image has digits with confidence
        pebbleDigitsCrops, pebbleDigitBoxes, pebbleDigitScores, goodPredictions, goodMasks, originalDigitCrops = digit_area_detection(
            frame)

        # see if digits were detected
        if pebbleDigitsCrops is not None:
            print('Frame with digits:', str(frameNumber))
            # tag and update pebble data
            # we create a new pebble if new detection
            if self.activePebble is None:
                # create new pebble
                self.numOfPebbles += 1
                self.activePebble = Pebble(
                    self.numOfPebbles, frameNumber, videoTime)
            self.pebbleLastSeen = frameNumber

            # update boxes
            self.activePebble.addDigitBoxes(
                pebbleDigitBoxes, frameNumber, videoTime)

            # check if converged already
            if not self.activePebble.isConverged:
                # save orientation bar prediction
                for i in range(len(pebbleDigitsCrops)):
                    resizedDigitCrop = square_and_resize(
                        pebbleDigitsCrops[i], 750)
                    annImg, fixedImages = digit_area_orientation_alignment(
                        resizedDigitCrop, originalDigitCrops[i])
                    cv2.imwrite(os.path.join(self.imgFolder, "DAOA_" +
                                str(frameNumber) + "_pred_"+str(i)+".jpg"), annImg)
                    for f in range(len(fixedImages)):
                        # downsize image
                        downsizedImage = fixedImages[f]
                        scale_percent = 25  # percent of original size
                        width = int(
                            downsizedImage.shape[1] * scale_percent / 100)
                        height = int(
                            downsizedImage.shape[0] * scale_percent / 100)
                        dim = (width, height)

                        downsizedImage = cv2.resize(
                            downsizedImage, dim, interpolation=cv2.INTER_AREA)
                        # prediciton
                        predImg, predlabels, predScores = digit_recognition(
                            downsizedImage)
                        if predImg is not None:
                            cv2.imwrite(os.path.join(self.imgFolder, "img_" +
                                        str(frameNumber) + "_pred_"+str(f)+".jpg"), predImg)
                            # update digits
                            self.activePebble.addDigits(
                                predlabels, predScores)
        # create frame based on current active pebbles
        frameWithData = addToFrame(og_frame, self, frameNumber, videoTime)
        # put frame into video
        self.processed_video.write(frameWithData)


def addToFrame(frame, video, frameNumber, videoTime):
    height, width = frame.shape[:2]
    pebble = video.activePebble
    if pebble is not None:
        # check if detected this frame
        currentClassification = pebble.obtainFinalClassification()
        if pebble.lastSeen == frameNumber:
            # add in digit detection area
            if pebble.currentDigitBoxes is not None:
                for digitBox in pebble.currentDigitBoxes:
                    minCord = (digitBox[0], digitBox[1])
                    maxCord = (digitBox[2], digitBox[3])
                    cv2.rectangle(frame, minCord, maxCord,
                                  color=(0, 255, 255), thickness=3)
            # reset current boxes
            pebble.resetBoxes()
        # setup text
        predText = None
        color = None
        if pebble.isConverged:
            predText = 'Final Identification: ' + \
                str(currentClassification)
            color = (6, 219, 88)
        else:
            predText = 'Highest Confidence: ' + \
                str(currentClassification)
            color = (238, 0, 242)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # get boundary of this text
        textsize = cv2.getTextSize(predText, font, 8, 15)[0]

        # get coords based on boundary
        textX = int(width - textsize[0]) / 2

        cv2.putText(frame, predText, (int(textX), 325),
                    cv2.FONT_HERSHEY_SIMPLEX, 8, color, thickness=15)

    # add in time
    cv2.putText(frame, str(round(videoTime, 2))+'s', (50, 125), cv2.FONT_HERSHEY_SIMPLEX,
                5, (225, 255, 0), thickness=10)
    return frame

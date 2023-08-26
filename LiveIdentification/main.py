import threading
import sys
import os
import cv2
import numpy as np
import torch
import sys_path
import training_utilities.transforms as T
import time
from video_util import Video
# ensure we are running on the correct gpu
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('GPU not being used, exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class MainProcessor:
    def __init__(self):

        # this array will hold pebbles we identify from the inlet stream
        self.inletPebbles = []
        # this array will hold pebbles we identify from the outlet stream but cannot match with a pebble saved from the inlet stream
        self.outletUnrecognizedPebbles = []
        # this array will save identified pebbles
        self.identifiedPebbles = []
        # we need to lock the arrays to prevent simultaneous access
        self.lock = threading.Lock()

    # function for each video
    def identify(self, videoname, isInlet):
        save_folder = f"./"+videoname+" Results/"
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        # create video
        video = Video(videoname, save_folder, isInlet)
        # set frames count and fps
        num_frames = video.frame_count
        FPS = video.fps

        start = time.time()

        frameNumber = 0
        hasFrames, frame = video.vidcap.read()
        while hasFrames:
            print(videoname, ' processing frame #', frameNumber)
            videoTime = frameNumber/FPS
            # process inlet frame
            video.processNextFrame(
                frame, frameNumber, videoTime)

            # process any identified pebbles
            video.processPebble(frameNumber, self.inletPebbles,
                                self.outletUnrecognizedPebbles, self.identifiedPebbles, self.lock)
            # check if we are currently processing
            # if none in frame can skip
            if video.activePebble is None or video.activePebble.isConverged:
                # skip four frames
                for i in range(4):
                    hasFrames, frame = video.vidcap.read()
                    frameNumber += 1
            hasFrames, frame = video.vidcap.read()
            frameNumber += 1

        end = time.time()
        print('Total time elapsed:', (end-start))
        print("Videoname: ", videoname)

        # When everything done, release the capture
        video.vidcap.release()
        video.processed_video.release()
        cv2.destroyAllWindows()


inletVideoName = 'CARVED~1'
# outletVideoName = 'Outlet Example'

# create main processor
mp = MainProcessor()

# create threads
inletThread = threading.Thread(target=mp.identify, args=(inletVideoName, True))
# outletThread = threading.Thread(
#     target=mp.identify, args=(outletVideoName, False))

# start threads
inletThread.start()
# outletThread.start()

inletThread.join()
# outletThread.join()


# display results
print('Identified Pebbles: ', mp.identifiedPebbles)
print('Inlet Unrecognized: ', mp.inletPebbles)
print('Outlet Unrecognized: ', mp.outletUnrecognizedPebbles)

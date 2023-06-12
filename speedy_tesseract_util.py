import cv2
import pytesseract
from pytesseract import Output
import os
import torch
import sys
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
    score = None
    # convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(n_boxes):
        text = "".join(d["text"][i]).strip()
        conf = int(d["conf"][i])
        if conf > 0.95:
            if not had_pred:
                pred = text
                had_pred = True
                score = conf
            (x, y, w, h) = (d['left'][i], d['top']
                            [i], d['width'][i], d['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(img, str(text), (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        4, (0, 255, 0), thickness=10)
    if not had_pred:
        cv2.putText(img, 'NONE', (5, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), thickness=10)

        return img, pred, score

    return img, pred, score


# [1, 1, 0, 4, 1, 4, 1, 3, 1, 0, 2, 2, 3, 3, 1, 7, 5, 11, 5, 7, 6, 2, 5, 3, 10, 3, 1, 1, 1, 2, 3, 4, 7, 5, 4, 3, 7, 6, 6, 5, 4, 8, 6, 5]

# [2, 2, 1, 4, 5, 2, 3, 1, 2, 3, 3, 2, 4, 7, 8, 6, 10, 10, 15, 8, 12, 4, 4, 11, 7, 11, 11, 7, 5, 6, 8, 6, 5, 3, 6, 9, 7, 6, 14, 11, 8, 6, 12, 10, 14, 10, 2]

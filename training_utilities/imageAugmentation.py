import cv2
import numpy as np
import random


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_NEAREST)

    # resize to original dimensions
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle,  cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int 
        height of the image

    w : int 
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  

    Returns 
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(image, -1, k)


def rotateImageAndBoxes(img_tfm, boxes):
    w, h = img_tfm.shape[1], img_tfm.shape[0]
    cx, cy = w//2, h//2

    # obtain degree to rotate by
    randRot = random.randint(0, 3)
    # direction of rotation
    coin = random.randint(0, 1)
    if coin == 1:
        randRot = 360 - randRot
        # print('FLIPPED!!', randRot)

    # rotate image
    img_tfm = rotate_im(img_tfm, randRot)

    # get four courners of boxes
    corners = get_corners(boxes)
    # combine into one
    corners = np.hstack((corners, boxes[:, 4:]))
    # rotate boxes
    corners[:, :8] = rotate_box(corners[:, :8], randRot, cx, cy, h, w)
    # select minimum point and maximum point
    new_bbox = corners[:, [0, 1, 6, 7]]

    scale_factor_x = img_tfm.shape[1] / w

    scale_factor_y = img_tfm.shape[0] / h

    img_tfm = cv2.resize(img_tfm, (w, h))

    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y,
                        scale_factor_x, scale_factor_y]

    bboxes = new_bbox

    return img_tfm, bboxes

def augmentImageAndBoxes(img_tfm, boxes):
    w, h = img_tfm.shape[1], img_tfm.shape[0]
    cx, cy = w//2, h//2

    # obtain degree to rotate by
    randRot = random.randint(0, 3)
    # direction of rotation
    coin = random.randint(0, 1)
    if coin == 1:
        randRot = 360 - randRot
        # print('FLIPPED!!', randRot)
    # # Gaussian blur
    rand1 = random.randint(1, 50)
    if rand1 % 2 == 0:
        rand1 += 1
    rand2 = random.randint(1, 50)
    if rand2 % 2 == 0:
        rand2 += 1
    img_tfm = cv2.GaussianBlur(img_tfm, (rand1, rand2), 0)

    # motion blur
    img_tfm = apply_motion_blur(img_tfm, 10, random.randint(1, 360))

    # rotate image
    img_tfm = rotate_im(img_tfm, randRot)

    # get four courners of boxes
    corners = get_corners(boxes)
    # combine into one
    corners = np.hstack((corners, boxes[:, 4:]))
    # rotate boxes
    corners[:, :8] = rotate_box(corners[:, :8], randRot, cx, cy, h, w)
    # select minimum point and maximum point
    new_bbox = corners[:, [0, 1, 6, 7]]

    scale_factor_x = img_tfm.shape[1] / w

    scale_factor_y = img_tfm.shape[0] / h

    img_tfm = cv2.resize(img_tfm, (w, h))

    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y,
                        scale_factor_x, scale_factor_y]

    bboxes = new_bbox

    return img_tfm, bboxes


def sharpen_digits(img):
    # convert image to grayscale
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)

    # get blurred image
    blur = cv2.blur(img, (10, 10))
    # subtract from original to get mask
    mask = img - blur
    # mix with sharpened image
    sharpened = img + 4*mask

    sharpened[sharpened > 255] = 255
    sharpened[sharpened < 0] = 0
    backtorgb = cv2.cvtColor(np.uint8(sharpened), cv2.COLOR_GRAY2RGB)

    return backtorgb


def brighten_digits(img):
    # convert image to grayscale
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
    # threshold to binary
    thresh = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)[1]
    # add blurring to thresholded image
    thresh = cv2.GaussianBlur(thresh, (25, 25), 0)
    # mix in brightened thresholded image to original
    img = img + 0.4*thresh
    # normalize
    img = img/1.4

    # sharpen image
    # kernel = np.array([[-1,-1,-1],
    #                     [-1, 9,-1],
    #                     [-1,-1,-1]])
    # sharpened = cv2.filter2D(img, -1, kernel)

    # # sharpen
    # blur = cv2.blur(img, (10, 10))
    # mask = img - blur
    # img_sharpened = img + 4.5*mask

    # # convert back to rgb
    # img_sharpened[img_sharpened > 255] = 255
    # img_sharpened[img_sharpened < 0] = 0

    # backtorgb = cv2.cvtColor(np.uint8(img_sharpened), cv2.COLOR_GRAY2RGB)

    # return backtorgb

    # convert back to rgb
    # img[img > 255] = 255
    backtorgb = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB)

    return backtorgb


def augmentImage(img_tfm):
    # # Gaussian blur
    rand1 = random.randint(1, 50)
    if rand1 % 2 == 0:
        rand1 += 1
    rand2 = random.randint(1, 50)
    if rand2 % 2 == 0:
        rand2 += 1
    img_tfm = cv2.GaussianBlur(img_tfm, (rand1, rand2), 0)

    # motion blur
    img_tfm = apply_motion_blur(img_tfm, 10, random.randint(1, 360))

    return img_tfm


def augmentImageAndRotate(img_tfm):
    w, h = img_tfm.shape[1], img_tfm.shape[0]
    cx, cy = w//2, h//2

    # obtain degree to rotate by
    randRot = random.randint(0, 3)
    # direction of rotation
    coin = random.randint(0, 1)
    if coin == 1:
        randRot = 360 - randRot
        # print('FLIPPED!!', randRot)
    # # Gaussian blur
    rand1 = random.randint(1, 50)
    if rand1 % 2 == 0:
        rand1 += 1
    rand2 = random.randint(1, 50)
    if rand2 % 2 == 0:
        rand2 += 1
    img_tfm = cv2.GaussianBlur(img_tfm, (rand1, rand2), 0)

    # motion blur
    img_tfm = apply_motion_blur(img_tfm, 10, random.randint(1, 360))

    # rotate image
    img_tfm = rotate_im(img_tfm, randRot)

    # resize image to original w and h
    img_tfm = cv2.resize(img_tfm, (w, h))

    return img_tfm


def randomRotation(img1, img2):
    # apply the same random rotation to both images
    randRot = random.randint(0, 359)
    # rotate image 1
    img1 = rotate_im(img1, randRot)
    # rotate image 2
    img2 = rotate_im(img2, randRot)

    return img1, img2

# Function to map each intensity level to output intensity level.


def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2


def contrast_stretch(img):
    # Define parameters.
    r1 = 70
    s1 = 0
    r2 = 140
    s2 = 255

    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)

    # Apply contrast stretching.
    contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)

    return np.uint8(contrast_stretched)


def gamma_correct(img, gamma):
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype='uint8')
    return gamma_corrected

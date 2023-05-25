import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image, ImageEnhance, ImageOps
from xml.dom.minidom import parse
from train_utils.engine import train_one_epoch, evaluate
import train_utils.utils as utils
import train_utils.transforms as T
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from imageAugmentation import randomRotation, augmentImage

# ensure we are running on the correct gpu
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image filenames
        self.imgs = list(
            sorted(os.listdir(os.path.join(root, "SegImages"))))
        # load all mask filenames
        self.masks = list(
            sorted(os.listdir(os.path.join(root, "SegMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "SegImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "SegMasks", self.masks[idx])

        # open image and mask
        img = cv2.imread(img_path)
        # mask = Image.open(mask_path)
        # applying grayscale method
        # mask = ImageOps.grayscale(mask)
        # mask = np.array(mask)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # apply random rotation
        img, mask = randomRotation(img, mask)
        # randomly augment image
        img = augmentImage(img)
        # convert np img to PIL img
        img = Image.fromarray(img).convert("RGB")
        # randomly augment brightness
        bright_dec = random.uniform(0.2, 1.5)
        enhancer = ImageEnhance.Brightness(img)
        # to reduce brightness by 50%, use factor 0.5
        img = enhancer.enhance(bright_dec)

        boxes = []
        labels = []
        masks = []
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # print(obj_ids)
        # split the color-encoded mask into a set
        # of binary masks
        binMasks = mask == obj_ids[:, None, None]

        for i in range(len(binMasks)):
            binMask = binMasks[i]

            # get bounding box coordinates binMask
            pos = np.where(binMask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            masks.append(binMask)
            if obj_ids[i] != 0:
                labels.append(2)
            else:
                labels.append(1)

        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # # lets try to understand what our box and mask look like
        # def get_coloured_mask(mask, i, masks_len):
        #     """
        #     random_colour_masks
        #     parameters:
        #         - image - predicted masks
        #     method:
        #         - the masks of each predicted object is given random colour for visualization
        #     """
        #     r = np.zeros_like(mask).astype(np.uint8)
        #     g = np.zeros_like(mask).astype(np.uint8)
        #     b = np.zeros_like(mask).astype(np.uint8)
        #     if i == 0:
        #         c = [0, 0, 255]
        #     elif i == 1:
        #         c = [255, 0, 0]
        #     elif i == 2:
        #         c = [0, 255, 0]
        #     elif i == 3:
        #         c = [255, 255, 255]
        #     r[mask == 1], g[mask == 1], b[mask == 1] = c
        #     coloured_mask = np.stack([r, g, b], axis=2)
        #     return coloured_mask
        # rect_th = 2
        # text_size = 2
        # text_th = 2

        # toPIL = transforms.ToPILImage()
        # img_with_annot = np.asarray(toPIL(img))

        # for i in range(len(masks)):
        #     rgb_mask = get_coloured_mask(target["masks"][i], i, len(masks))
        #     # print('Annot:', img_with_annot.shape)
        #     # print(rgb_mask.shape)
        #     img_with_annot = cv2.addWeighted(
        #         img_with_annot, 1, rgb_mask, 0.5, 0)
        #     cv2.rectangle(img_with_annot, (int(target["boxes"][i][0]), int(target["boxes"][i][1])), (int(target["boxes"][i][2]), int(target["boxes"][i][3])),
        #                   color=(0, 255, 0), thickness=rect_th)
        #     cv2.putText(img_with_annot, str(labels[i].item()), (int(target["boxes"][i][0]), int(target["boxes"][i][1])), cv2.FONT_HERSHEY_SIMPLEX,
        #                 text_size, (0, 255, 0), thickness=text_th)

        # # create folder to save pebbles
        # vis_tgt_path = "./GroupDetectionFrameMulti/GroupSegExamples/"
        # if not os.path.isdir(vis_tgt_path):
        #     os.mkdir(vis_tgt_path)
        # cv2.imwrite(os.path.join(vis_tgt_path, "sample_" +
        #             str(idx) + ".jpg"), img_with_annot)

        return img, target

    def __len__(self):
        return len(self.imgs)


def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Stop here if you are fine-tunning Faster-RCNN

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


dataroot = './GroupDetectionFrameMulti/'
# use our dataset and defined transformations
dataset = SegmentationDataset(dataroot, get_transform(train=True))
dataset_test = SegmentationDataset(dataroot, get_transform(train=False))

# # split the dataset in train and test set
# torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()
# num_train = int(0.25 * len(indices))
# dataset = torch.utils.data.Subset(dataset, indices[:num_train])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])

# print("Number in train dataset:", len(indices[:num_train]))
# print("Number in test dataset:", len(indices[num_train:]))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 3

# get the model using our helper function
model = build_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# define the optimizer, here we use SGD
optimizer = torch.optim.SGD(
    params, lr=0.003, momentum=0.9, weight_decay=0.0005)

# define a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2)

# optimizer = torch.optim.SGD(params, lr=0.005,
#                             momentum=0.9, weight_decay=0.0005)

# # and a learning rate scheduler which decreases the learning rate by
# # 10x every 3 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

# number of epochs
num_epochs = 200

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader,
                    device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)
    if epoch == 100:
        torch.save(model, './saved_models/group-seg-multi-100.pt')

evaluate(model, data_loader_test, device=device)

model_save = './saved_models/'
if not os.path.isdir(model_save):
    os.mkdir(model_save)

torch.save(model, './saved_models/group-seg-multi.pt')

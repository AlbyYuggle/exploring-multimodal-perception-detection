import os
import random

import cv2
import numpy as np

import torch
import torch.utils.data as DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from src.config import VOC_IMG_MEAN
import glob

def get_train_test():
    data_dir = "./data/**/*.npy"
    file_names = glob.glob(data_dir)
    file_names = [[f.split('_boxes_')[0]+"_cam_"+f.split('_boxes_')[1][:-3]+"jpg", f, f.split('_boxes_')[0]+"_depth_"+f.split('_boxes_')[1][:-3]+"jpg"] for f in file_names]
    split = int(len(file_names) * 0.997)
    train = file_names[:split]
    test = file_names[split:]
    return VocDetectorDataset(train, S=14), VocDetectorDataset2(test, S=14)


def pad(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )

class VocDetectorDataset(DataLoader.Dataset):
    

    def __init__(
        self,
        files,
        S=14,
        encode_target=True,
    ):
        print("Initializing dataset")
        self.transform = [transforms.ToTensor()]
        self.boxes = []
        self.labels = []
        self.mean = VOC_IMG_MEAN
        self.S = S
        self.image_size = (448, 448)
        self.encode_target = encode_target

        self.files = files

    def __getitem__(self, idx):
        fname = self.files[idx]
        camf = fname[0]
        boxesf = fname[1]
        depthf = fname[2]
        img = cv2.imread(camf, cv2.IMREAD_COLOR)
        dimg = cv2.imread(depthf, cv2.IMREAD_GRAYSCALE)
        dimg_small = torch.nn.MaxPool2d((4,4), stride=(4,4))(torch.unsqueeze(torch.unsqueeze(torch.tensor(dimg, dtype=float), axis=0), axis=0)).repeat((1,3,1,1))
        dimg_small = dimg_small[0].permute(1,2,0)
        dimg_small = cv2.resize(dimg_small.numpy(), self.image_size)
        lll = np.load(boxesf)
        boxes = torch.Tensor(lll[:, :4])
        labels = lll[:, 4]-1
        if len(labels.shape) == 2:
            labels = np.squeeze(labels)
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = cv2.resize(img, self.image_size)
        # img = cv2.cvtColor(
        #     img, cv2.COLOR_BGR2RGB
        # )  # because pytorch pretrained model use RGB
        img = img - np.array(
            self.mean, dtype=np.float32
        )  # subtract dataset mean image (in RGB format)

        target_boxes, target_cls, has_object_map = self.encoder(
            boxes, labels
        )  # SxSx(B*5+C)

        for t in self.transform:
            img = t(img)
            dimg_small = t(dimg_small)

        return img, target_boxes, target_cls, has_object_map, dimg_small

    def __len__(self):
        return len(self.files)

    def encoder(self, boxes, labels):
        """
        This function takes as input bounding boxes and corresponding labels for a particular image
        sample and outputs a target tensor of size SxSx(5xB+C)

        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return SxSx(5xB+C) (14x14x30 in our case)
        """
        grid_num = self.S
        target = torch.zeros((grid_num, grid_num, 8))
        cell_size = 1.0 / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        center_xy_all = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(center_xy_all.size()[0]):
            center_xy = center_xy_all[i]
            ij = (center_xy / cell_size).ceil() - 1
            # confidence represents iou between predicted and ground truth
            target[int(ij[1]), int(ij[0]), 4] = 1  # confidence of box 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
            xy = ij * cell_size  # coordinates of upper left corner
            delta_xy = (center_xy - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy

        target_boxes = target[:, :, :4]
        has_object_map = (target[:, :, 4:5]) > 0
        has_object_map = has_object_map.squeeze()
        target_cls = target[:, :, 5:]

        return target_boxes, target_cls, has_object_map
    
class VocDetectorDataset2(DataLoader.Dataset):
    

    def __init__(
        self,
        files,
        S=14,
        encode_target=True,
    ):
        print("Initializing dataset")
        self.transform = [transforms.ToTensor()]
        self.boxes = []
        self.labels = []
        self.mean = VOC_IMG_MEAN
        self.S = S
        self.image_size = (448, 448)
        self.encode_target = encode_target

        self.files = files

    def __getitem__(self, idx):
        fname = self.files[idx]
        camf = fname[0]
        boxesf = fname[1]
        depthf = fname[2]
        img = cv2.imread(camf, cv2.IMREAD_COLOR)
        dimg = cv2.imread(depthf, cv2.IMREAD_GRAYSCALE)
        dimg_small = torch.nn.MaxPool2d((4,4), stride=(4,4))(torch.unsqueeze(torch.unsqueeze(torch.tensor(dimg, dtype=float), axis=0), axis=0)).repeat((1,3,1,1))
        dimg_small = dimg_small[0].permute(1,2,0)
        dimg_small = cv2.resize(dimg_small.numpy(), self.image_size)
        lll = np.load(boxesf)
        boxes = torch.Tensor(lll[:, :4])
        labels = (np.log2(lll[:, 4])+0.25).astype(int)
        if len(labels.shape) == 2:
            labels = np.squeeze(labels)
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = cv2.resize(img, self.image_size)
        # img = cv2.cvtColor(
        #     img, cv2.COLOR_BGR2RGB
        # )  # because pytorch pretrained model use RGB
        img = img - np.array(
            self.mean, dtype=np.float32
        )  # subtract dataset mean image (in RGB format)

        target_boxes, target_cls, has_object_map = self.encoder(
            boxes, labels
        )  # SxSx(B*5+C)

        for t in self.transform:
            img = t(img)
            dimg_small = t(dimg_small)

        return img, target_boxes, target_cls, has_object_map, dimg_small, boxes, labels, fname[0]

    def __len__(self):
        return len(self.files)

    def encoder(self, boxes, labels):
        """
        This function takes as input bounding boxes and corresponding labels for a particular image
        sample and outputs a target tensor of size SxSx(5xB+C)

        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return SxSx(5xB+C) (14x14x30 in our case)
        """
        grid_num = self.S
        target = torch.zeros((grid_num, grid_num, 8))
        cell_size = 1.0 / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        center_xy_all = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(center_xy_all.size()[0]):
            center_xy = center_xy_all[i]
            ij = (center_xy / cell_size).ceil() - 1
            # confidence represents iou between predicted and ground truth
            target[int(ij[1]), int(ij[0]), 4] = 1  # confidence of box 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
            xy = ij * cell_size  # coordinates of upper left corner
            delta_xy = (center_xy - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy

        target_boxes = target[:, :, :4]
        has_object_map = (target[:, :, 4:5]) > 0
        has_object_map = has_object_map.squeeze()
        target_cls = target[:, :, 5:]

        return target_boxes, target_cls, has_object_map

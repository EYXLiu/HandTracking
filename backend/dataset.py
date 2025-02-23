import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import json

class HandmarkDatabase(Dataset):
    def __init__(self, img_path, mark_path, transform=None):
        self.img_path = img_path
        with open(os.path.join(mark_path, "_annotations.coco.json"), "r") as f:
            coco = json.load(f)
        data = {img["id"]: img["file_name"] for img in coco["images"]}
        self.annotations = []
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            filename = data.get(image_id)
            keypoints = torch.tensor(ann["keypoints"], dtype=torch.float32)
            keypoints = keypoints[torch.arange(keypoints.size(0)) % 3 != 2]
            full_img_path = os.path.join(self.img_path, filename)
            self.annotations.append({"image_path": full_img_path, "keypoints": keypoints})

        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        image_path, keypoints = sample["image_path"], sample["keypoints"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, keypoints, image_path
    

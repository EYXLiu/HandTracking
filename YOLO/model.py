import torch
import torch.nn as nn
import torch.functional as F 

class YOLO(nn.Modele):
    def __init__(self, num_classes, S, B):
        super (YOLO, self).__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.SiLU(), 
            nn.MaxPool2d(2, 2)
        )
        
        self.detect = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128 * 7 * 7, 4096), 
            nn.SiLU(), 
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes))
        )
        
    def forward(self, x):
        x = self.cnn(x), 
        x = self.detect(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x
    
    
def YOLO_loss(predictions, targets, S, B, num_classes):
    lambda_coord = 5
    lambda_noobj = 0.5
    
    obj_mask = targets[...,4] > 0
    noobj_mask = ~obj_mask
    
    coord_loss = lambda_coord * torch.sum((predictions[...,:2] - targets[...,:2]) ** 2) # MSE for bounding box
    obj_loss = torch.sum((predictions[obj_mask, 4] - targets[obj_mask, 4]) ** 2) # MSE for confidence of object
    noobj_loss = lambda_noobj * torch.sum((predictions[noobj_mask, 4] - targets[noobj_mask, 4]) ** 2) # MSE if theres no object
    class_loss = F.cross_entropy(predictions[...,5:], targets[...,5:])
    
    return coord_loss + obj_loss + noobj_loss + class_loss
    
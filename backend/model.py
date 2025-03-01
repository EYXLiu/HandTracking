import torch
import torch.nn as nn
from torch.nn import functional as F

class VisionCNN(nn.Module):
    def __init__(self, config):
        super(VisionCNN, self).__init__()
        self.config = config
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.input_layers, config.filter_1, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(config.filter_1, config.filter_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(config.filter_2, config.filter_3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.linear_layers = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(config.linear_flatten, config.linear_layer),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.linear_layer, config.output_layer),
        )
        
    def forward(self, x, targets=None):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        x = x.sigmoid()
        x = x * 224
        
        loss = None
        if targets is not None:
            loss = F.mse_loss(x, targets)
        
        return x, loss
    
    def predict(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        x = x.sigmoid()
        x = x * 224
        
        return x

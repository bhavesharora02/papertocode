import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.conv2d1 = nn.Conv2d(**{'in_channels': 3, 'out_channels': 64, 'kernel_size': 7, 'stride': 2})
        
        
        
        self.maxpool2d2 = nn.MaxPool2d(**{'kernel_size': 3, 'stride': 2})
        
        
        
        self.seq3 = nn.Sequential(
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(64),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(64),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(64),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq4 = nn.Sequential(
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.BatchNorm2d(128),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(128),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(128),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(128),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq5 = nn.Sequential(
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq6 = nn.Sequential(
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq7 = nn.Sequential(
            
            nn.AdaptiveAvgPool2d((1,1)),
            
            nn.Flatten(),
            
            nn.Linear(512, 1000),
            
        )
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        if x is None:
            x = input_ids
        out = x
        
        
        out = self.conv2d1(out)
        
        
        
        out = self.maxpool2d2(out)
        
        
        
        out = self.seq3(out)
        
        
        
        out = self.seq4(out)
        
        
        
        out = self.seq5(out)
        
        
        
        out = self.seq6(out)
        
        
        
        out = self.seq7(out)
        
        
        return out
        
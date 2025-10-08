import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.seq1 = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            
            nn.BatchNorm2d(64),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.maxpool2d2 = nn.MaxPool2d(**{'kernel_size': 3, 'stride': 2, 'padding': 1})
        
        
        
        self.seq3 = nn.Sequential(
            
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq4 = nn.Sequential(
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(512),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq5 = nn.Sequential(
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(1024),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq6 = nn.Sequential(
            
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.BatchNorm2d(2048),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(2048),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(2048),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.adaptiveavgpool2d7 = nn.AdaptiveAvgPool2d(**{'output_size': [1, 1]})
        
        
        
        self.seq8 = nn.Sequential(
            
            nn.Flatten(),
            
            nn.Linear(2048, 1000),
            
        )
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        if x is None:
            x = input_ids
        out = x
        
        
        out = self.seq1(out)
        
        
        
        out = self.maxpool2d2(out)
        
        
        
        out = self.seq3(out)
        
        
        
        out = self.seq4(out)
        
        
        
        out = self.seq5(out)
        
        
        
        out = self.seq6(out)
        
        
        
        out = self.adaptiveavgpool2d7(out)
        
        
        
        out = self.seq8(out)
        
        
        return out
        
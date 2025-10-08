import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.seq1 = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq2 = nn.Sequential(
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.maxpool2d3 = nn.MaxPool2d(**{'kernel_size': 2, 'stride': 2, 'padding': 0})
        
        
        
        self.seq4 = nn.Sequential(
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq5 = nn.Sequential(
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.maxpool2d6 = nn.MaxPool2d(**{'kernel_size': 2, 'stride': 2, 'padding': 0})
        
        
        
        self.seq7 = nn.Sequential(
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq8 = nn.Sequential(
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq9 = nn.Sequential(
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq10 = nn.Sequential(
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.maxpool2d11 = nn.MaxPool2d(**{'kernel_size': 2, 'stride': 2, 'padding': 0})
        
        
        
        self.seq12 = nn.Sequential(
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq13 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq14 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq15 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.maxpool2d16 = nn.MaxPool2d(**{'kernel_size': 2, 'stride': 2, 'padding': 0})
        
        
        
        self.seq17 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq18 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq19 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.seq20 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.maxpool2d21 = nn.MaxPool2d(**{'kernel_size': 2, 'stride': 2, 'padding': 0})
        
        
        
        self.seq22 = nn.Sequential(
            
            nn.Flatten(),
            
            nn.Linear(25088, 4096),
            
        )
        
        
        
        self.dropout23 = nn.Dropout(**{'p': 0.5})
        
        
        
        self.seq24 = nn.Sequential(
            
            nn.Flatten(),
            
            nn.Linear(4096, 4096),
            
        )
        
        
        
        self.dropout25 = nn.Dropout(**{'p': 0.5})
        
        
        
        self.seq26 = nn.Sequential(
            
            nn.Flatten(),
            
            nn.Linear(4096, 1000),
            
        )
        
        
        
        self.softmax27 = nn.Softmax(**{})
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        if x is None:
            x = input_ids
        out = x
        
        
        out = self.seq1(out)
        
        
        
        out = self.seq2(out)
        
        
        
        out = self.maxpool2d3(out)
        
        
        
        out = self.seq4(out)
        
        
        
        out = self.seq5(out)
        
        
        
        out = self.maxpool2d6(out)
        
        
        
        out = self.seq7(out)
        
        
        
        out = self.seq8(out)
        
        
        
        out = self.seq9(out)
        
        
        
        out = self.seq10(out)
        
        
        
        out = self.maxpool2d11(out)
        
        
        
        out = self.seq12(out)
        
        
        
        out = self.seq13(out)
        
        
        
        out = self.seq14(out)
        
        
        
        out = self.seq15(out)
        
        
        
        out = self.maxpool2d16(out)
        
        
        
        out = self.seq17(out)
        
        
        
        out = self.seq18(out)
        
        
        
        out = self.seq19(out)
        
        
        
        out = self.seq20(out)
        
        
        
        out = self.maxpool2d21(out)
        
        
        
        out = self.seq22(out)
        
        
        
        out = self.dropout23(out)
        
        
        
        out = self.seq24(out)
        
        
        
        out = self.dropout25(out)
        
        
        
        out = self.seq26(out)
        
        
        
        out = self.softmax27(out)
        
        
        return out
        
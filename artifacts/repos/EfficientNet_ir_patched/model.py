import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.lazyconv2d1 = nn.LazyConv2d(**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'out_channels': 16})
        
        
        
        self.seq2 = nn.Sequential(
            
            nn.LazyConv2d(16, kernel_size=3, padding=1, bias=False),
            
            nn.BatchNorm2d(16),
            
            nn.ReLU(inplace=True),
            
        )
        
        
        
        self.lazyconv2d3 = nn.LazyConv2d(**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'out_channels': 16})
        
        
        
        self.adaptiveavgpool2d4 = nn.AdaptiveAvgPool2d(**{'output_size': (1, 1)})
        
        
        
        self.dropout5 = nn.Dropout(**{'p': 0.2})
        
        
        
        self.seq6 = nn.Sequential(
            
            nn.Flatten(),
            
            nn.LazyLinear(10),
            
        )
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        if x is None:
            x = input_ids
        out = x
        
        
        out = self.lazyconv2d1(out)
        
        
        
        out = self.seq2(out)
        
        
        
        out = self.lazyconv2d3(out)
        
        
        
        out = self.adaptiveavgpool2d4(out)
        
        
        
        out = self.dropout5(out)
        
        
        
        out = self.seq6(out)
        
        
        return out
        
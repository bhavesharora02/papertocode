import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.seq1 = nn.Sequential(
            
            nn.Flatten(),
            
            nn.LazyLinear(128),
            
            nn.ReLU(),
            
            nn.Linear(128, 2),
            
        )
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        if x is None:
            x = input_ids
        out = x
        
        
        out = self.seq1(out)
        
        
        return out
        
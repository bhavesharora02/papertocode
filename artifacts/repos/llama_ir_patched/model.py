import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.transformer1 = nn.Transformer(**{'d_model': 8192, 'nhead': 64, 'num_encoder_layers': 80})
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        out = x
        
        out = self.transformer1(out)
        
        return out
        
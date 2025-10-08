import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.transformerencoder1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, activation='gelu', batch_first=True), num_layers=24)
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        if x is None:
            x = input_ids
        out = x
        
        
        out = self.transformerencoder1(out.view(out.size(0), -1, 1024))
        
        
        return out
        
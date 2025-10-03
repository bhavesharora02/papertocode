import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.encoder = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3
        )
        
        
        
        self.linear2 = nn.Linear(**{"in_features": 768, "out_features": 3})
        
        
        
        self.softmax3 = nn.Softmax(**{})
        
        

    def forward(self, input_ids=None, attention_mask=None, labels=None, x=None):
        
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
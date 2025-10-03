import torch

def accuracy(outputs, labels):
    if isinstance(outputs, torch.Tensor):
        if outputs.ndim > 1:  # logits
            preds = torch.argmax(outputs, dim=-1)
        else:  # already predicted classes
            preds = outputs
    else:
        preds = outputs  # in case already int list/ndarray
    return (preds == labels).float().mean().item()

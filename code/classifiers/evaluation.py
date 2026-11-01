"""
Shared model-building and inference helpers of the classifier scripts.

    get_model(architecture, num_classes, device)    untrained ResNet-50 or Swin-B with a new head
    evaluate_model(model, dataloader, device, ...)  probabilities, predictions and labels
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def get_model(architecture, num_classes, device):
    """Instantiates the correct architecture with an untrained head."""
    if architecture == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'swin_b':
        model = models.swin_b()
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model.to(device)

def evaluate_model(model, dataloader, device, num_classes):
    """Runs inference and collects predictions, probabilities, and true labels."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs =[]
    
    with torch.no_grad():
        for inputs, labels in dataloader: # Removed inner tqdm to keep logs clean
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0)
    )

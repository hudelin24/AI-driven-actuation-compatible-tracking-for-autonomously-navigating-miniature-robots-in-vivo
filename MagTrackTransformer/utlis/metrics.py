import torch
import numpy as np

def l1_error(preds, labels):
    """
    Args:
        preds (array): array of predictions. Dimension is bsz * output_dim.
        labels (array): array of labels. Dimension is bsz * output_dim.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    return torch.mean(torch.abs(preds-labels))

def euclidean_error(preds, labels):
    """
    Args:
        preds (array): array of predictions. Dimension is bsz * output_dim.
        labels (array): array of labels. Dimension is bsz * output_dim.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    
    diff = (preds - labels) ** 2
    euclidean_err = torch.sum(diff, -1) ** 0.5
    
    return torch.mean(euclidean_err)

def angular_error(preds, labels):
    """
    Args:
        preds (array): array of predictions. Dimension is bsz * output_dim.
        labels (array): array of labels. Dimension is bsz * output_dim.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    cos_sim = torch.sum(preds*labels, dim=1)
    #print(torch.mean(torch.abs(preds-labels)))
    cos_sim = torch.clamp(cos_sim,-1.0,1.0)
    angle_rad = torch.acos(cos_sim)
    angle_deg = angle_rad * 180 / torch.pi

    return torch.mean(torch.abs(angle_deg))
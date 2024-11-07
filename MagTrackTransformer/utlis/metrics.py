import torch

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
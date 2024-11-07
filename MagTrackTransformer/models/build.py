import torch
from models.model_registry import MODEL_REGISTRY
import models.mat



def build_model(cfg):
    """
    Builds the magnetic model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone.
    """

    # Construct the model
    name = cfg.MODEL_NAME
    gpu_id = cfg.GPU_ID
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.GPU_ENABLE:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)

    return model


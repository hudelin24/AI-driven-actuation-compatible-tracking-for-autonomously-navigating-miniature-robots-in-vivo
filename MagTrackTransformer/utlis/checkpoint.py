import os 
import torch
from fvcore.common.file_io import PathManager
from utlis.logging import get_logger

logger = get_logger(__name__)

def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if not PathManager.exists(checkpoint_dir):
        try:
            PathManager.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir

def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")

def get_path_to_checkpoint(path_to_job, epoch):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)

def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = PathManager.ls(d) if PathManager.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)

def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = PathManager.ls(d) if PathManager.exists(d) else []
    return any("checkpoint" in f for f in files)

def is_checkpoint_epoch(cfg, cur_epoch):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (CfgNode): configs to save.
        cur_epoch (int): current number of epoch of the model.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True

    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0

def save_checkpoint(path_to_job, model, optimizer, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """
    # Ensure that the checkpoint dir exists.
    PathManager.mkdirs(get_checkpoint_dir(path_to_job))

    sd = model.state_dict()

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
    with PathManager.open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint

def load_checkpoint(
    path_to_checkpoint,
    model,
    optimizer=None,
    epoch_reset=False):
    """
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        optimizer (optim): optimizer to load the historical state.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert PathManager.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with PathManager.open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    pre_train_dict = checkpoint["model_state"]
    model_dict = model.state_dict()

    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k
        for k in model_dict.keys()
        if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            logger.info("Network weights {} not loaded.".format(k))
    # Load pre-trained weights.
    model.load_state_dict(pre_train_dict_match, strict=False)
    epoch = -1

    # Load the optimizer state (commonly not done when fine-tuning)
    if "epoch" in checkpoint.keys() and not epoch_reset:
        epoch = checkpoint["epoch"]
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        epoch = -1
    return epoch

def load_test_checkpoint(cfg, model):
    """
    Loading checkpoint logic for testing.
    """
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in MODEL_VIS.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TEST.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            path_to_checkpoint=cfg.TEST.CHECKPOINT_FILE_PATH,
            model=model,
            optimizer=None,
            epoch_reset=False,
        )
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        load_checkpoint(
            path_to_checkpoint=last_checkpoint,
            model=model,
            optimizer=None,
            epoch_reset=False,
        )
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            path_to_checkpoint=cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model=model,
            optimizer=None,
            epoch_reset=False,
        )
    else:
        logger.info(
            "Unknown way of loading checkpoint. Using with random initialization."
        )

        load_checkpoint(
            path_to_checkpoint=cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model=model,
            optimizer=None,
            epoch_reset=False,
        )

def load_train_checkpoint(cfg, model, optimizer):
    """
    Loading checkpoint logic for training.
    """
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch =  load_checkpoint(
                                path_to_checkpoint=last_checkpoint,
                                model=model,
                                optimizer=optimizer,
                                epoch_reset=False,
                            )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch =  load_checkpoint(
                                path_to_checkpoint=cfg.TRAIN.CHECKPOINT_FILE_PATH,
                                model=model,
                                optimizer=optimizer,
                                epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                            )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    return start_epoch


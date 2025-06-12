import pprint
import torch
import torch.nn as nn
import os
import numpy as np
import utlis.logging as logging
import utlis.misc as misc
import utlis.optimizer as optim
import utlis.checkpoint as cu
import datasets.loader as loader
from utlis.meters import TrainMeter, ValMeter
import utlis.metrics as metrics
from models.build import build_model
import visualization.tensorboard_vis as tb
import utlis.losses as losses

logger = logging.get_logger(__name__)




def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the magnetic training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model: the tracking model to train
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. 
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    data_size = len(train_loader)
    cur_device = next(model.parameters()).device

    for cur_iter, (mag_map_s, cam_data) in enumerate(train_loader):
        #print(mag_map_s.shape, cam_data.shape)

        # Transfer the data to the current GPU device.
        train_meter.iter_timer.reset()
        if cfg.DATA_AUGUMENTATION:
            mag_map_s, cam_data = misc.data_augument(mag_map_s, cam_data)
        
        #print(mag_map_s.shape, cam_data.shape)

        if cfg.GPU_ENABLE:
            mag_map_s = mag_map_s.to(cur_device, non_blocking=True)         #[bsz, in_chans, T, H, W]
            cam_data = cam_data.to(cur_device, non_blocking=True)           #[bsz, 3]
        
        bsz = mag_map_s.shape[0]

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL_MTT.LOSS_FUNC)(reduction="mean")

        preds = model((mag_map_s[:,3:,:,:,:] - mag_map_s[:,0:3,:,:,:]).reshape(bsz,-1))       #[bsz, 3]
        loss = loss_fun(preds, cam_data)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        #nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.5)
        optimizer.step()

        l1_err = metrics.l1_error(preds, cam_data)
        euclidean_err = metrics.euclidean_error(preds, cam_data)
        
        # Copy the stats from GPU to CPU (sync point).
        loss, l1_err, euclidean_err = (
                                        loss.item(),
                                        l1_err.item(),
                                        euclidean_err.item(),
                                    )
        # Update and log stats.
        train_meter.update_stats(
            l1_err,
            euclidean_err,
            loss,
            lr,
            bsz,
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/l1_err": l1_err,
                    "Train/euclidean_err": euclidean_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_timer.pause()
        train_meter.log_iter_stats(cur_epoch, cur_iter)

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model: the tracking model to train
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. 
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    data_size = len(val_loader)
    cur_device = next(model.parameters()).device

    for cur_iter, (mag_map_s, cam_data) in enumerate(val_loader):
        val_meter.iter_timer.reset()

        # Transfer the data to the current GPU device.
        if cfg.GPU_ENABLE:
            mag_map_s = mag_map_s.to(cur_device, non_blocking=True)         #[bsz, in_chans, T, H, W]
            cam_data = cam_data.to(cur_device, non_blocking=True)           #[bsz, 3]

        bsz = mag_map_s.shape[0]
        preds = model((mag_map_s[:,3:,:,:,:] - mag_map_s[:,0:3,:,:,:]).reshape(bsz,-1))       #[bsz, 3]

        l1_err = metrics.l1_error(preds, cam_data)
        euclidean_err = metrics.euclidean_error(preds, cam_data)
                        
        # Copy the stats from GPU to CPU (sync point).
        l1_err, euclidean_err = (
                                    l1_err.item(),
                                    euclidean_err.item(),
                                )
        
        # Update and log stats.
        val_meter.update_stats(
            l1_err,
            euclidean_err,
            bsz,
        )

        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "val/l1_err": l1_err,
                    "val/euclidean_err": euclidean_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        val_meter.update_predictions(preds, cam_data)
        val_meter.iter_timer.pause()
        val_meter.log_iter_stats(cur_epoch, cur_iter)

    # Log epoch stats.
    l1_err_epoch = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    return l1_err_epoch


def train(cfg):
    """
    Train an mlp for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. 
    """ 
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train MLP with config:")
    logger.info(pprint.pformat(cfg))

    # Build the magnetic tracking model.
    model = build_model(cfg)

    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model)


    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable (mtt).
    if not cfg.TRAIN.FINETUNE:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
        start_epoch = 0
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)


    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE:
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    best_l1_err = float("inf")

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        
        is_eval_epoch = misc.is_eval_epoch(
            cfg, 
            cur_epoch,
        )

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            l1_err_epoch = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

            # Save the model with smallest val l1_err
            if l1_err_epoch < best_l1_err:
                best_l1_err = l1_err_epoch
                best_epoch = cur_epoch
    
    logger.info("Best epoch: {}".format(best_epoch + 1))

    if writer is not None:
        writer.close()

    return os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_epoch_{:05d}.pyth".format(best_epoch + 1))
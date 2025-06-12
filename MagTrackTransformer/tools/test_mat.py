import torch
import numpy as np
import utlis.logging as logging
import utlis.misc as misc
import utlis.checkpoint as cu
import datasets.loader as loader
from utlis.meters import TestMeter as TestMeter_pos
from utlis.meters import TestMeter_orient as TestMeter_ori
import utlis.metrics as metrics
from models.build import build_model

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter_pos, test_meter_ori, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        test_loader (loader): data loader to provide testing data.
        model (model): model to evaluate the performance.
        test_meter_pos (TestMeter): meter instance to record and calculate the metrics.
        test_meter_ori (TestMeter): meter instance to record and calculate the metrics.
        cfg (CfgNode): configs. 
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    cur_device = next(model.parameters()).device


    for cur_iter, (mag_map_s, cam_pos, cam_ori) in enumerate(test_loader):

        if cfg.GPU_ENABLE:
            mag_map_s = mag_map_s.to(cur_device, non_blocking=True)         #[bsz, in_chans, T, H, W]
            cam_pos = cam_pos.to(cur_device, non_blocking=True)             #[bsz, 3]
            cam_ori = cam_ori.to(cur_device, non_blocking=True)             #[bsz, 3]

        bsz = mag_map_s.shape[0]
        preds = model(mag_map_s.reshape(bsz, mag_map_s.shape[1], mag_map_s.shape[2],
                            mag_map_s.shape[3]*mag_map_s.shape[4]))

        l1_err = metrics.l1_error(preds[0], cam_pos)
        euclidean_err = metrics.euclidean_error(preds[0], cam_pos)
        angular_err = metrics.angular_error(preds[1], cam_ori)

        
        # Copy the stats from GPU to CPU (sync point).
        l1_err, euclidean_err = (
                                l1_err.item(),
                                euclidean_err.item(),
                            )

        angular_err = angular_err.item()
            
            
        # Update and log stats.
        test_meter_pos.update_stats(
            l1_err,
            euclidean_err,
            bsz,
        )

        test_meter_ori.update_stats(
            angular_err,
            bsz,
        )
        test_meter_pos.log_iter_stats(cur_iter)
        test_meter_pos.update_predictions(preds[0], cam_pos)
        test_meter_ori.log_iter_stats(cur_iter)
        test_meter_ori.update_predictions(preds[1], cam_ori)

    #Save preds.
    test_meter_pos.save_predictions()
    test_meter_pos.log_stats()
    test_meter_pos.reset()

    test_meter_ori.save_predictions()
    test_meter_ori.log_stats()
    test_meter_ori.reset()


def test(cfg):
    """
    Args:
        cfg (CfgNode): configs. 
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test Magnetic Actuation Transformer with config:")
    logger.info(cfg)

    # Build the magnetic tracking model and print model statistics.
    model = build_model(cfg)

    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model)

    cu.load_test_checkpoint(cfg, model)

    # Create the testing loader.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create the testing meter.
    test_meter_pos = TestMeter_pos(len(test_loader), cfg)
    test_meter_ori = TestMeter_ori(len(test_loader), cfg)

    perform_test(test_loader, model, test_meter_pos, test_meter_ori, cfg)

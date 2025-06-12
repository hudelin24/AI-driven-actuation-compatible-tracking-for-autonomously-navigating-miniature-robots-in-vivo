import torch
import numpy as np
import utlis.logging as logging
import utlis.misc as misc
import utlis.checkpoint as cu
import datasets.loader as loader
from utlis.meters import TestMeter_orient as TestMeter
import utlis.metrics as metrics
from models.build import build_model

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        test_loader (loader): data loader to provide testing data.
        model (model): model to evaluate the performance.
        test_meter (TestMeter): meter instance to record and calculate the metrics.
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

        cam_pos_save = cam_pos.detach()
        bsz = mag_map_s.shape[0]
        if cfg.MODEL_MOT.MAG_NOISE_ENABLE:
            mag_map_s = misc.add_mag_noise(cfg, mag_map_s)

        if cfg.MODEL_MOT.AUXILIARY_NOISE_ENABLE:
            cam_pos = misc.add_auxiliary_noise(cfg, cam_pos)

        if cfg.MODEL_MOT.AUXILIARY_ENABLE:
            preds = model(mag_map_s.reshape(bsz, mag_map_s.shape[1], mag_map_s.shape[2],
                            mag_map_s.shape[3]*mag_map_s.shape[4]), cam_pos.unsqueeze(1))        
        else:
            preds = model(mag_map_s.reshape(bsz, mag_map_s.shape[1], mag_map_s.shape[2],
                            mag_map_s.shape[3]*mag_map_s.shape[4]))

        angular_err = metrics.angular_error(preds, cam_ori)

        
        # Copy the stats from GPU to CPU (sync point).
        angular_err = angular_err.item()
            
            
        # Update and log stats.
        test_meter.update_stats(
            angular_err,
            bsz,
        )
        test_meter.log_iter_stats(cur_iter)
        test_meter.update_predictions(preds, torch.cat([cam_pos_save,cam_ori], 1))

    #Save preds.
    test_meter.save_predictions()
    test_meter.log_stats()
    test_meter.reset()

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
    logger.info("Test Magnetic Orientation Transformer with config:")
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
    test_meter = TestMeter(len(test_loader), cfg)

    perform_test(test_loader, model, test_meter, cfg)

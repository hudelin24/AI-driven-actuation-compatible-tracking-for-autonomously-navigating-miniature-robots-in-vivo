import torch
import numpy as np
import utlis.logging as logging
import utlis.misc as misc
import utlis.checkpoint as cu
import datasets.loader as loader
from utlis.meters import TestMeter
import utlis.metrics as metrics
from models.build import build_model

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, magnetic_noise, model, test_meter, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        test_loader (loader): data loader to provide testing data.
        magnetic_noise (Tensor): the magnetic noise to add to data.
        model (model): model to evaluate the performance.
        test_meter (TestMeter): meter instance to record and calculate the metrics.
        cfg (CfgNode): configs. 
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    cur_device = next(model.parameters()).device


    for cur_iter, (mag_map) in enumerate(test_loader):

        if cfg.GPU_ENABLE:
            mag_map = mag_map.to(cur_device, non_blocking=True)         #[bsz, in_chans, T, num_mag]
        mag_map_noise_free = mag_map[:,:,0,:].clone()

        bsz, in_chans, T, num_mag = mag_map.shape[0], mag_map.shape[1], mag_map.shape[2], mag_map.shape[3]
        # Add noise to the magnetic map.
        mag_map, gt_noise = misc.add_post_calib_noise(cfg, mag_map, magnetic_noise)

        preds = model(mag_map)                                          #[bsz, num_mag, in_chans]
    

        l1_err = metrics.l1_error(preds, gt_noise)
        euclidean_err = metrics.euclidean_error(preds, gt_noise)

        
        l1_err, euclidean_err = (
                                    l1_err.item(),
                                    euclidean_err.item(),
                                )
            
            
        # Update and log stats.
        test_meter.update_stats(
            l1_err,
            euclidean_err,
            bsz,
        )
        test_meter.log_iter_stats(cur_iter)
        test_meter.update_predictions(preds, torch.cat([gt_noise, mag_map_noise_free.permute(0,2,1)],-1))

    #Save preds.
    test_meter.save_predictions()

    # Log stats.
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
    logger.info("Test Magnetic Denoise Transformer with config:")
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

    # Load noise.
    magnetic_noise = torch.load(cfg.MODEL_MDT.POST_CALIB_NOISE_DIR)

    perform_test(test_loader, magnetic_noise, model, test_meter, cfg)

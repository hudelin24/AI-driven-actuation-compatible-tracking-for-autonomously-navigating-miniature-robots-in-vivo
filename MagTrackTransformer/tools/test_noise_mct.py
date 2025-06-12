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
def perform_test(test_loader, magnetic_noise, model, model_MDT, test_meter, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        test_loader (loader): data loader to provide testing data.
        magnetic_noise (Tensor): the magnetic noise to add to data.
        model (model): model to evaluate the performance.
        model_MDT: Trained MDT model.
        test_meter (TestMeter): meter instance to record and calculate the metrics.
        cfg (CfgNode): configs. 
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    model_MDT.eval()
    cur_device = next(model.parameters()).device

    for cur_iter, (mag_map) in enumerate(test_loader):

        H, W = cfg.MODEL_MDT.TSUS_SIZE[-2], cfg.MODEL_MDT.TSUS_SIZE[-1]
        num_calib = cfg.MODEL_MDT.CSUS_SIZE[-1]
        bsz, in_chans, T, num_mag = mag_map.shape[0], mag_map.shape[1], mag_map.shape[2], mag_map.shape[3]
        # Transfer the data to the current GPU device.
        if cfg.GPU_ENABLE:
            mag_map = mag_map.to(cur_device, non_blocking=True)                                           #[bsz, in_chans, T, num_mag]
        mag_map_s = mag_map[:,:,0,0:H*W].reshape(bsz, in_chans, 1, H, W).clone()                          #[bsz, in_chans, 1, H, W]
        mag_map_c = mag_map[:,:,0,H*W:].unsqueeze(2).clone()                                              #[bsz, in_chans, 1, num_calib]
        # Add noise to the magnetic map.
        mag_map, gt_noise = misc.add_post_calib_noise(cfg, mag_map, magnetic_noise)
        mag_map_c_noise = mag_map[:,:,0,H*W:].reshape(bsz, in_chans, 1, num_calib)
        
        #Denoise by MDT
        pred_noise = model_MDT(mag_map).detach()                                                          #[bsz, num_mag, in_chans]
        mag_map_denoise = (mag_map[:,:,0,:] - pred_noise.permute(0,2,1)).unsqueeze(2)                     #[bsz, inchans, 1, num_mag]
        mag_map_s_denoise = mag_map_denoise[:,:,0,0:H*W].reshape(bsz, in_chans, 1, H, W)
        mag_map_c_denoise = mag_map_denoise[:,:,0,H*W:].reshape(bsz, in_chans, 1, num_calib)

        preds = model(mag_map_c)                                                                          #[bsz, H, W, in_chans]
        preds_noise = model(mag_map_c_noise)                                                                           
        preds_denoise = model(mag_map_c_denoise)                                                                           

        l1_err = metrics.l1_error(preds_denoise, mag_map_s.squeeze(2).permute(0,2,3,1))
        euclidean_err = metrics.euclidean_error(preds_denoise, mag_map_s.squeeze(2).permute(0,2,3,1))

        # Copy the stats from GPU to CPU (sync point).
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
        test_meter.update_predictions(torch.cat([preds, preds_noise, preds_denoise], -1), mag_map_s.squeeze(2).permute(0,2,3,1))

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
    logger.info("Test Magnetic Calibration Tranformer with config:")
    logger.info(cfg)

    # Build the magnetic calibration model and print model statistics.
    cfg.MODEL_NAME = "mct_base"
    model = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model)
    cu.load_test_checkpoint(cfg, model)

    # Build the magnetic denoise model and print model statistics.
    cfg.MODEL_NAME = "mdt_base"
    model_MDT = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model_MDT)
    cu.load_checkpoint(cfg.MODEL_MDT.CHECKPOINT_FILE_PATH, model_MDT)

    # Create the testing loader.
    test_loader = loader.construct_loader(cfg, "test")
    #logger.info("Number of samples in test set: {}", len(test_loader))

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create the testing meter.
    test_meter = TestMeter(len(test_loader), cfg)

    # Load noise.
    magnetic_noise = torch.load(cfg.MODEL_MDT.POST_CALIB_NOISE_DIR)

    perform_test(test_loader, magnetic_noise, model, model_MDT, test_meter, cfg)

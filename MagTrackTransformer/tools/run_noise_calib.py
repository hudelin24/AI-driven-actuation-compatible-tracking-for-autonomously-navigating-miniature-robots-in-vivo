import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import torch
import numpy as np
import utlis.logging as logging
import utlis.misc as misc
import utlis.checkpoint as cu
import datasets.loader as loader
from utlis.meters import TrackingNoiseRawDataMeter
from models.build import build_model
from utlis.parser import load_config, parse_args

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_calib(data_loader, model_MCT, model_MDT, meter, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        data_loader (loader): data loader to provide raw tracking data.
        model_MCT: Trained MCT model.
        model_MDT: Trained MDT model.
        meter (TestMeter): meter instance to record preds and .
        cfg (CfgNode): configs. 
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model_MCT.eval()
    model_MDT.eval()
    cur_device = next(model_MCT.parameters()).device
    magnetic_noise = torch.load(cfg.MODEL_MDT.POST_CALIB_NOISE_DIR)

    for _, (mag_map, path_to_mag) in enumerate(data_loader):
        #print(mag_map.shape, path_to_mag)                                      #[bsz, in_chans, T, H*W+num_calib]
        H, W = cfg.MODEL_MDT.TSUS_SIZE[-2], cfg.MODEL_MDT.TSUS_SIZE[-1]
        num_calib = cfg.MODEL_MDT.CSUS_SIZE[-1]
        bsz, in_chans, T, num_mag = mag_map.shape[0], mag_map.shape[1], mag_map.shape[2], mag_map.shape[3]
        # Transfer the data to the current GPU device.
        if cfg.GPU_ENABLE:
            mag_map = mag_map.to(cur_device, non_blocking=True)                                           #[bsz, in_chans, T, num_mag]
        mag_map_s = mag_map[:,:,0,0:H*W].reshape(bsz, in_chans, 1, H, W).clone()                          #[bsz, in_chans, 1, H, W]
        mag_map_c = mag_map[:,:,0,H*W:].unsqueeze(2).clone()                                              #[bsz, in_chans, 1, num_calib]
        # Add noise to the magnetic map.
        if cfg.MODEL_MDT.POST_CALIB_ENABLE:
            mag_map, gt_noise = misc.add_post_calib_noise(cfg, mag_map, magnetic_noise)
        mag_map_s_noise = mag_map[:,:,0,0:H*W].reshape(bsz, in_chans, 1, H, W).clone()
        mag_map_c_noise = mag_map[:,:,0,H*W:].reshape(bsz, in_chans, 1, num_calib)
        
        #Denoise by MDT
        pred_noise = model_MDT(mag_map).detach()                                                          #[bsz, num_mag, in_chans]
        mag_map_denoise = (mag_map[:,:,0,:] - pred_noise.permute(0,2,1)).unsqueeze(2)                     #[bsz, inchans, 1, num_mag]
        mag_map_s_denoise = mag_map_denoise[:,:,0,0:H*W].reshape(bsz, in_chans, 1, H, W)
        mag_map_c_denoise = mag_map_denoise[:,:,0,H*W:].reshape(bsz, in_chans, 1, num_calib)
                
        preds = model_MCT(mag_map_c)                                                                       #[bsz, H, W, in_chans]
        preds_noise = model_MCT(mag_map_c_noise)                                                                           
        preds_denoise = model_MCT(mag_map_c_denoise)                                                                           

        tracking_mag_map = torch.cat((mag_map_s.cpu(), mag_map_s_noise.cpu(), 
                                    mag_map_s_denoise.cpu(), preds.cpu().permute(0,3,1,2).unsqueeze(2), 
                                    preds_noise.cpu().permute(0,3,1,2).unsqueeze(2)
                                    , preds_denoise.cpu().permute(0,3,1,2).unsqueeze(2)),1)                 #[bsz, in_chans * 6, 1, H, W]
        meter.update_data(tracking_mag_map, path_to_mag)
    meter.save_data()
    meter.reset()


def calib(cfg):
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
    logger.info("Calib the noise tracking data with the pretrained Magnetic Calibration Transormer and Magnetic Denoise Transformer:")
    logger.info(cfg)

    # Build the magnetic calibration model and print model statistics.
    model_MCT = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model_MCT)
    cu.load_test_checkpoint(cfg, model_MCT)

    # Build the magnetic denoise model and print model statistics.
    cfg.MODEL_NAME = "mdt_base"
    model_MDT = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model_MDT)
    cu.load_checkpoint(cfg.MODEL_MDT.CHECKPOINT_FILE_PATH, model_MDT)



    for split in ["train"]:
        data_loader = loader.construct_loader(cfg, split)
        meter = TrackingNoiseRawDataMeter(cfg)
        perform_calib(data_loader, model_MCT, model_MDT, meter, cfg)

def main():
    """
    Main function 
    """
    args = parse_args()
    cfg = load_config(args)
    calib(cfg)

if __name__ == "__main__":
    main()


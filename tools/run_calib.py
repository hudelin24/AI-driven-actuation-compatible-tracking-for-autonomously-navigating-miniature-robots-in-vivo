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
from utlis.meters import TrackingRawDataMeter
from models.build import build_model
from utlis.parser import load_config, parse_args

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_calib(data_loader, model, meter, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        data_loader (loader): data loader to provide raw tracking data.
        model (model): model to evaluate the performance.
        meter (TestMeter): meter instance to record preds and .
        cfg (CfgNode): configs. 
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    cur_device = next(model.parameters()).device

    for _, (
        mag_map_c, mag_map_s, path_to_mag
                ) in enumerate(data_loader):

        # Transfer the data to the current GPU device.
        if cfg.GPU_ENABLE:
            mag_map_c = mag_map_c.to(cur_device, non_blocking=True)
                
        preds = model(mag_map_c)                         #[bsz,4,7,3]
        tracking_mag_map = torch.cat((preds.clone().detach().cpu().permute(0,3,1,2).unsqueeze(2), 
                                        mag_map_s),1)              # [bsz, in_chans * 2, 1, H, W]

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
    logger.info("Calib the raw tracking data with the pretrained Magnetic Calibration Transormer:")
    logger.info(cfg)

    # Build the magnetic calibration model and print model statistics.
    model = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model)

    cu.load_test_checkpoint(cfg, model)

    # Create the testing loader.
    for split in ["train"]:
        data_loader = loader.construct_loader(cfg, split)
        meter = TrackingRawDataMeter(cfg)
        perform_calib(data_loader, model, meter, cfg)

def main():
    """
    Main function 
    """
    args = parse_args()
    cfg = load_config(args)
    calib(cfg)

if __name__ == "__main__":
    main()


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
def perform_test(test_loader, model, cfg):
    """
    Evaluate the model on the testing set.
    Args:
        test_loader (loader): data loader to provide testing data.
        model (model): model to evaluate the performance.
        cfg (CfgNode): configs. 
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    cur_device = next(model.parameters()).device
    preds = []

    for cur_iter, (mag_map_s) in enumerate(test_loader):
        if cfg.GPU_ENABLE:
            mag_map_s = mag_map_s.to(cur_device, non_blocking=True)         #[bsz, in_chans, T, H, W]

                
        preds.append(model(mag_map_s[:,3:,:,:,:] - mag_map_s[:,0:3,:,:,:]).cpu())       #[bsz, 3]

    torch.save(preds, os.path.join(cfg.OUTPUT_DIR, "preds.pyth"))



def navig(cfg):
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
    logger.info(cfg)

    # Build the magnetic calibration model and print model statistics.
    model = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model)

    cu.load_test_checkpoint(cfg, model)

    # Create the testing loader.
    for split in ["test"]:
        data_loader = loader.construct_loader(cfg, split)
        perform_test(data_loader, model, cfg)



def main():
    """
    Main function 
    """
    args = parse_args()
    cfg = load_config(args)
    navig(cfg)

if __name__ == "__main__":
    main()


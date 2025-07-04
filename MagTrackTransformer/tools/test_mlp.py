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
    mag_map_s_list = []
    cam_data_list = []


    for cur_iter, (mag_map_s, cam_data) in enumerate(test_loader):
        #print(mag_map_s.shape, cam_data.shape)

        if cfg.GPU_ENABLE:
            mag_map_s = mag_map_s.to(cur_device, non_blocking=True)         #[bsz, in_chans, T, H, W]
            cam_data = cam_data.to(cur_device, non_blocking=True)           #[bsz, 3]

                
        bsz = mag_map_s.shape[0]
        preds = model((mag_map_s[:,3:,:,:,:] - mag_map_s[:,0:3,:,:,:]).reshape(bsz,-1))       #[bsz, 3]
        mag_map_s_list.append((mag_map_s[:,3:,0,:,:] - mag_map_s[:,0:3,0,:,:]).clone().detach().cpu())
        cam_data_list.append(cam_data.clone().detach().cpu())

        l1_err = metrics.l1_error(preds, cam_data)
        euclidean_err = metrics.euclidean_error(preds, cam_data)

        
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
        test_meter.update_predictions(preds, cam_data)

    #Save preds.
    #test_meter.save_predictions()
    test_meter.log_stats()
    test_meter.reset()
    print(torch.cat(mag_map_s_list).shape)
    print(torch.cat(cam_data_list).shape)
    torch.save(torch.cat(mag_map_s_list), cfg.OUTPUT_DIR + "/Mag_map_s.pyth")
    torch.save(torch.cat(cam_data_list), cfg.OUTPUT_DIR + "/Cam_data.pyth")


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
    logger.info("Test Magnetic Tracking Transformer with config:")
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

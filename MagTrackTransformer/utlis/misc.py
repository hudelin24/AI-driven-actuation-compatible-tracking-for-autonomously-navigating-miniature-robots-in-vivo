from utlis.logging import get_logger
import torch
import os
import numpy as np 
import math
import datetime
from einops import rearrange


logger = get_logger(__name__)

def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3

def log_model_info(model):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")

def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. 
        cur_epoch (int): current epoch.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0

def perform_calib(cfg, model, mag_map_c):
    """
    Args:
        cfg (CfgNode): configs. 
        model: trained mct.
        mag_map_c:              [bsz, in_chans, T, num_calib]
    
    Return:
        preds:                  [bsz, in_chans, T, H, W]
    
    """
    bsz, in_chans, T, num_calib = mag_map_c.shape
    step = cfg.MAX_NUM_CALIB_SAMPLE
    if T > 1:
        mag_map_c = rearrange(mag_map_c.permute(0,2,1,3), 'b t c d -> (b t) c d', b=bsz, t=T, c=in_chans, d=num_calib).unsqueeze(2) #[bsz * T, in_chans, 1, num_calib]
        #print("misc mag_map_c", mag_map_c.shape)
    
    if bsz * T > step:
        preds = [model(mag_map_c[i:i+step]).detach() for i in range(0, bsz * T, step)]   #[step, H, W, in_chans]
        preds = torch.cat(preds)                                                         #[bsz, H, W, inchans]
    else:
        preds = model(mag_map_c)

    preds = preds.permute(0,3,1,2)
    H, W = preds.shape[-2], preds.shape[-1]

    if T > 1:
        preds = rearrange(preds, '(b t) c h w -> b t c h w', b=bsz, t=T, c=in_chans, h=H, w=W)
        preds = preds.permute(0,2,1,3,4)
    else:
        preds = preds.unsqueeze(2)
    

    return preds            #[bsz, in_chans, T, H, W]

def add_noise(cfg, mag_map_c, mag_map_s):
    """
    Args:
        cfg (CfgNode): configs. 
        mag_map_c:              [bsz, in_chans, T, num_calib]
        mag_map_s:              [bsz, in_chans, T, H, W]    
    """
    bsz, in_chans, T, H, W = mag_map_s.shape
    num_calib = mag_map_c.shape[-1]
    
    t_c = (torch.arange(T) / cfg.SAMPLING_FREQ).expand(bsz, in_chans, num_calib, T).permute(0,1,3,2)
    amp_c = torch.randn(bsz, in_chans, 1, num_calib)
    phase_c = torch.randn(bsz, in_chans, 1, num_calib) * np.pi * 2
    freq_c = torch.randn(bsz, in_chans, 1, num_calib) * 4 + 48
    t_c = 2 * np.pi * freq_c * t_c + phase_c
    noise_c = amp_c * torch.sin(t_c)
    mag_map_c = mag_map_c + noise_c 
    
    t_s = (torch.arange(T) / cfg.SAMPLING_FREQ).expand(bsz, in_chans, H, W, T).permute(0,1,4,2,3)
    amp_s = torch.randn(bsz, in_chans, 1, H, W)
    phase_s = torch.randn(bsz, in_chans, 1, H, W) * np.pi * 2
    freq_s = torch.randn(bsz, in_chans, 1, H, W) * 4 + 48
    t_s = 2 * np.pi * freq_s * t_s + phase_s
    noise_s = amp_s * torch.sin(t_s)
    mag_map_s = mag_map_s + noise_s 

    #print(noise_c.shape, noise_s.shape)

    return mag_map_c, mag_map_s

def data_augument(mag_map_s, cam_data):
    """
    Args:
        mag_map_s:                  [bsz, in_chans, T, H, W]
        cam_data:                   [bsz, 3]
    """
    bsz, in_chans, T, H, W = mag_map_s.shape
    noise = torch.normal(0, 0.05, mag_map_s.shape)
    mag_map_s_augument = mag_map_s[:,:,0,:,:].unsqueeze(2).expand(bsz, in_chans, T, H, W) + noise

    return torch.cat([mag_map_s, mag_map_s_augument], 0), torch.cat([cam_data, cam_data], 0)


def add_mag_noise(cfg, mag_map_s):
    """
    Args:
        cfg (CfgNode): configs. 
        mag_map_s:              [bsz, in_chans, T, H, W]    
    """
    bsz = mag_map_s.shape[0]
    mag_noise_distribution = torch.load(cfg.MODEL_MOT.MAG_NOISE_DIR)
    mag_noise = torch.randn(bsz, mag_map_s.shape[3], mag_map_s.shape[4], mag_map_s.shape[1])
    mag_noise = mag_noise * mag_noise_distribution[:,:,3:6].unsqueeze(0).expand(bsz, -1, -1, -1) + mag_noise_distribution[:,:,0:3].unsqueeze(0).expand(bsz, -1, -1, -1)
    mag_map_s = mag_map_s + mag_noise.permute(0,3,1,2).unsqueeze(2).to(mag_map_s.device, non_blocking=True)

    return mag_map_s


def add_auxiliary_noise(cfg, cam_pos):
    """
    Args:
        cfg (CfgNode): configs. 
        cam_pos:              [bsz, 3]    
    """
    bsz = cam_pos.shape[0]
    auxiliary_noise_distribution = torch.load(cfg.MODEL_MOT.AUXILIARY_NOISE_DIR).to(cam_pos.device, non_blocking=True)
    depth = torch.floor((0.182 - cam_pos[:,-1]) * 100).long() - 1
    auxiliary_noise_params = auxiliary_noise_distribution[depth]
    auxiliary_noise = torch.randn(bsz, cam_pos.shape[-1]).to(cam_pos.device, non_blocking=True)
    auxiliary_noise = auxiliary_noise * auxiliary_noise_params[:,3:6] + auxiliary_noise_params[:,0:3]
    cam_pos = cam_pos + auxiliary_noise

    return cam_pos

def add_post_calib_noise(cfg, mag_map, magnetic_noise):
    """
    Args:
        cfg (CfgNode): configs. 
        mag_map:              [bsz, in_chans, T, num_mag]
        magnetic_noise:       [N, num_mag, in_chans]    
    """
    bsz, in_chans, T, num_mag = mag_map.shape
    N = magnetic_noise.shape[0]
    num_noise_free_sample = torch.round(torch.tensor(cfg.MODEL_MDT.NOISE_FREE_RATIO * bsz * T)).to(torch.long)
    indices_noise_sample = torch.randperm(N)[:bsz*T]
    indices_noise_free_sample = torch.randperm(bsz*T)[:num_noise_free_sample]
    post_calib_noise = magnetic_noise[indices_noise_sample].to(mag_map.device, non_blocking=True)   
    post_calib_noise[indices_noise_free_sample] = 0.0                                       #[bsz*T, num_mag, in_chans]
    post_calib_noise = post_calib_noise.reshape(bsz,T,num_mag,in_chans).permute(0,3,1,2)    #[bsz, in_chans, T, num_mag]

    mag_map = mag_map + post_calib_noise

    
    return mag_map, post_calib_noise[:,:,0,:].permute(0,2,1)







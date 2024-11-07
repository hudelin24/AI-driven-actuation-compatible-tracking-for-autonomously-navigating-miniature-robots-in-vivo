import torch
import os
from torch.utils.data import Dataset
from fvcore.common.file_io import PathManager
import numpy as np
from datasets.dataset_registry import DATASET_REGISTRY
from utlis.logging import get_logger
import re

logger = get_logger(__name__)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

@DATASET_REGISTRY.register()
class CalibDataset(Dataset):
    def __init__(self, cfg, split):
        self.path_to_dataset = cfg.DATA.PATH_TO_DATA_DIR
        self.path_to_mag = []
        self.split = split

        refs = sorted([x for x in os.listdir(os.path.join(self.path_to_dataset,
                                             split)) if "ref" in x], key=natural_sort_key)

        for ref in refs:
            mag_files = sorted([os.path.join(self.path_to_dataset, split, ref, x) 
                            for x in os.listdir(os.path.join(self.path_to_dataset, 
                            split, ref)) if "mag_map" in x], key=natural_sort_key)            
            self.path_to_mag = self.path_to_mag + mag_files


        logger.info(split + " dataset includes {} samples.".format(len(self.path_to_mag)))


    def __len__(self):
        return len(self.path_to_mag)
    
    def __getitem__(self, index):
        
        with PathManager.open(self.path_to_mag[index], "rb") as f:
            target = torch.load(f, map_location="cpu")     
            mag_map_s = target["magnetic map sensing"]                        #[1,3,1,4,7]
            mag_map_c = target["magnetic map calibration"]                    #[1,3,1,12]
        
        mag_map_s = mag_map_s.squeeze(0)
        mag_map_c = mag_map_c.squeeze(0)

        return mag_map_c.float(), mag_map_s.float()


@DATASET_REGISTRY.register()
class TrackingRawDataset(Dataset):
    def __init__(self, cfg, split):
        self.path_to_dataset = cfg.DATA.PATH_TO_DATA_DIR
        self.path_to_mag = []
        
        refs = sorted([x for x in os.listdir(os.path.join(self.path_to_dataset,
                                             split)) if "ref" in x], key=natural_sort_key)

        for ref in refs:
            mag_files = sorted([os.path.join(self.path_to_dataset, split, ref, "mag_data", x) 
                            for x in os.listdir(os.path.join(self.path_to_dataset, 
                            split, ref, "mag_data")) if "mag" in x], key=natural_sort_key)            
            self.path_to_mag = self.path_to_mag + mag_files


        logger.info(split + " dataset includes {} samples.".format(len(self.path_to_mag)))

    def __len__(self):
        return len(self.path_to_mag)
    
    def __getitem__(self, index):
        with PathManager.open(self.path_to_mag[index], "rb") as f:
            target = torch.load(f, map_location="cpu")     
            mag_map_s = target["magnetic map sensing"]                        #[1,3,1,4,7]
            mag_map_c = target["magnetic map calibration"]                    #[1,3,1,12]

        mag_map_s = mag_map_s.squeeze(0)
        mag_map_c = mag_map_c.squeeze(0)

        return mag_map_c.float(), mag_map_s.float(), self.path_to_mag[index]


@DATASET_REGISTRY.register()
class TrackingDataset(Dataset):
    def __init__(self, cfg, split):
        self.path_to_dataset = cfg.DATA.PATH_TO_DATA_DIR
        self.path_to_time_sync = []
        self.time_window_size = cfg.MODEL_MTT.MAG_SIZE[1]
        
        refs = sorted([x for x in os.listdir(os.path.join(self.path_to_dataset, split)) if "ref" in x], key=natural_sort_key)

        for ref in refs:
            time_sync_files = sorted([os.path.join(self.path_to_dataset, split, ref, "time_synchronization", x) 
                            for x in os.listdir(os.path.join(self.path_to_dataset, 
                            split, ref, "time_synchronization")) if "time_synchronization" in x], key=natural_sort_key)            
            self.path_to_time_sync = self.path_to_time_sync + time_sync_files
        
        logger.info(split + " dataset includes {} samples.".format(len(self.path_to_time_sync)))

    def __len__(self):
        return len(self.path_to_time_sync)
    
    def __getitem__(self, index):
        
        mag_map_s = []
        with PathManager.open(self.path_to_time_sync[index], "rb") as f:
            time_sync = torch.load(f, map_location='cpu')
            index_c = time_sync["camera index"]
            index_m = time_sync["magnetic index"]
        
        path_to_ref = self.path_to_time_sync[index].split("time_synchronization")[0]
        path_to_cam = os.path.join(path_to_ref, "cam_data", "cam" + str(index_c) + ".pyth")

        with PathManager.open(path_to_cam, "rb") as f:
            cam = torch.load(f, map_location="cpu")
            cam_data = cam["camera data"]                                        #[3]                       
            cam_time = cam["camera time"]
        
        for i in range(self.time_window_size):
            #print(index_m)
            if index_m < i:
                print(self.path_to_time_sync[index],i)
            path_to_mag = os.path.join(path_to_ref, "mag_data", "mag" + str(index_m - i) + ".pyth")

            with PathManager.open(path_to_mag, "rb") as f:
                mag = torch.load(f, map_location="cpu")
                mag_map_s.append(mag["calibrated magetic map"])                     #[1,in_chans,1,H,W]
                if i == 0:
                    mag_time = mag["magnetic time"]
        
        mag_map_s = torch.cat(mag_map_s, 2)
        
        if np.abs(mag_time - cam_time) > 0.015:
            print(path_to_cam, path_to_mag, self.path_to_time_sync[index], mag_time - cam_time)
            raise ValueError("Synchronization Error")
        
        
        mag_map_s = mag_map_s.squeeze(0)
        #print(mag_map_s.shape, cam_data.shape)

        return mag_map_s.float(), cam_data.float()


@DATASET_REGISTRY.register()
class NavigationDataset(Dataset):
    def __init__(self, cfg, split):
        self.path_to_dataset = cfg.DATA.PATH_TO_DATA_DIR
        self.path_to_time_sync = []
        self.time_window_size = cfg.MODEL_MTT.MAG_SIZE[1]
        
        refs = sorted([x for x in os.listdir(os.path.join(self.path_to_dataset, split)) if "ref" in x], key=natural_sort_key)

        for ref in refs:
            time_sync_files = sorted([os.path.join(self.path_to_dataset, split, ref, "time_synchronization", x) 
                            for x in os.listdir(os.path.join(self.path_to_dataset, 
                            split, ref, "time_synchronization")) if "time_synchronization" in x], key=natural_sort_key)            
            self.path_to_time_sync = self.path_to_time_sync + time_sync_files
        
        logger.info(split + " dataset includes {} samples.".format(len(self.path_to_time_sync)))

    def __len__(self):
        return len(self.path_to_time_sync)
    
    def __getitem__(self, index):
        
        mag_map_s = []
        
        with PathManager.open(self.path_to_time_sync[index], "rb") as f:
            time_sync = torch.load(f, map_location='cpu')
            index_m = time_sync["magnetic index"]
        
        path_to_ref = self.path_to_time_sync[index].split("time_synchronization")[0]
        
        for i in range(self.time_window_size):
            path_to_mag = os.path.join(path_to_ref, "mag_data", "mag" + str(index_m - i) + ".pyth")

            with PathManager.open(path_to_mag, "rb") as f:
                mag = torch.load(f, map_location="cpu")
                mag_map_s.append(mag["calibrated magetic map"])                     #[1,in_chans,1,H,W]
        
        mag_map_s = torch.cat(mag_map_s, 2)
        mag_map_s = mag_map_s.squeeze(0)

        return mag_map_s.float()
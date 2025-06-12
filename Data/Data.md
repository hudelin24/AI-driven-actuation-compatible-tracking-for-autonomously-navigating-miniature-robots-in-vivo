# MCT_pretrain
Each `ref` folder corresponds to a sensor board placement. 
Each `mag_map.pyth` is a dictionary file which stores one sample. It includes two keys:

### 'magnetic map sensing'
A `[1,3,1,4,7]` float tensor records the readouts from the tracking sensing units (TSUs), where `3` is the number of measurement channels, and `4` and `7` represent the shape of the TSU array. 

### 'magnetic map calibration'
A `[1,3,1,12]` float tensor records the readouts from the calibration sensing units (CSUs), where `3` is the number of measurement channels, and `12` is the number of the CSUs.

# MCT_finetune
Each `calib_mtt_xxx` folder contains a calibration fine-tune dataset.
Each `mag_map.pyth` is a dictionary file which stores one sample. The data format is same with `MCT_pretrain`.

# MCT_calib
Each `calib_mtt_xxx` folder contains a tracking dataset which  shares the same environmental conditions as the calibration fine-tune dataset with the same folder name.

## cam_data/cam.pyth
`cam_data/cam.pyth` is a dictionary file which stores one frame of 3D camera data. It includes three keys:

### 'camera index'
An `int` number indicates the frame index in camera data.

### 'camera time'
A `float` number indicates the camera timestamp.

### 'camera data'
A `[3]` float tensor records the coordinates of the robot captured by the 3D camera.

## mag_data/mag.pyth
`mag_data/mag.pyth` is a dictionary file which stores one frame of sensor array data. It includes four keys before calibration and five keys after calibration:

### 'magnetic index'
An `int` number indicates the frame index in magnetic data.

### 'magnetic time'
A `float` number indicates the sensor array timestamp.

### 'magnetic map sensing'
A `[1,3,1,4,7]` float tensor records the readouts from the tracking sensing units (TSUs), where `3` is the number of measurement channels, and `4` and `7` rep/esent the shape of the TSU array. 

### 'magnetic map calibration'
A `[1,3,1,12]` float tensor records the readouts from the calibration sensing units (CSUs), where `3` is the number of measurement channels, and `12` is the number of the CSUs.

### 'calibrated magnetic map' (only available after MCT-based calibration)
A `[1,6,1,4,7]` float tensor records the readouts from the tracking sensing units (TSUs) and the output of the corresponding fine-tuned MCT, where `6` is the number of measurement channels `* 2`, and `4` and `7` represent the shape of the TSU array. 

### 'calibrated denoised magnetic map' (only available after MDT-based denoising and MCT-based calibration)
A `[1,18,1,4,7]` float tensor records noise-free, noisy and denoised TSUs' readouts (i.e., `[:,0:3,:,:,:]`, `[:,3:6,:,:,:]` and `[:,6:9,:,:,:]`) and the outputs of the corresponding fine-tuned MCT using noise-free, noisy and denoised CSUs' readouts as inputs (i.e., `[:,9:12,:,:,:]`, `[:,12:15,:,:,:]` and `[:,15:18,:,:,:]`). Here, `18` is the number of measurement channels `* 6`, and `4` and `7` represent the shape of the TSU array. 

## time_synchronization/time_synchronization.pyth
`time_synchronization/time_synchronization.pyth` is a dictionary file which stores time synchronization information. It includes four keys:

### 'camera index'
An `int` number indicates the frame index in camera data.

### 'magnetic index'
An `int` number indicates the frame index in magnetic data which is the nearest in time with the camera frame.

### 'camera time'
A `float` number indicates the camera timestamp.

### 'magnetic time'
A `float` number indicates the sensor array timestamp.

# MTT
It is actually the re-organized MCT_calib after MCT-based calibration.

# magnetic_noise.pyth
A `[1453849,40,3]` float tensor records the post-calibration noise in readouts of all sensing units, where `3` is the number of measurement channels, and `40` is the number of sensing units. 

# MTT_denoised
It is actually the re-organized MCT_calib after MDT-based denoising and MCT-based calibration.

# MOT
Each `ref` folder contains a segment of continuously collected data. 

## cam_data/cam.pyth
`cam_data/cam.pyth` is a dictionary file which stores one frame of 3D camera data. It includes four keys:

### 'camera index'
An `int` number indicates the frame index in camera data.

### 'camera time'
A `float` number indicates the camera timestamp.

### 'camera pos'
A `[3]` float tensor records the 3D coordinates of the robot captured by the 3D camera.

### 'camera ori'
A `[3]` float tensor records the unit orientation vector of the robot captured by the 3D camera.

## mag_data/mag.pyth
`mag_data/mag.pyth` is a dictionary file which stores one frame of sensor array data. It includes four keys before calibration and five keys after calibration:

### 'magnetic index'
An `int` number indicates the frame index in magnetic data.

### 'magnetic time'
A `float` number indicates the sensor array timestamp.

### 'magnetic map sensing'
A `[1,3,1,4,7]` float tensor records the readouts from the tracking sensing units (TSUs), where `3` is the number of measurement channels, and `4` and `7` rep/esent the shape of the TSU array. 

### 'magnetic map calibration'
A `[1,3,1,12]` float tensor records the readouts from the calibration sensing units (CSUs), where `3` is the number of measurement channels, and `12` is the number of the CSUs.

## time_synchronization/split(train or val or test)/time_synchronization.pyth
`time_synchronization/split(train or val or test)/time_synchronization.pyth` is a dictionary file which stores time synchronization information. It includes four keys:

### 'camera index'
An `int` number indicates the frame index in camera data.

### 'magnetic index'
An `int` number indicates the frame index in magnetic data which is the nearest in time with the camera frame.

### 'camera time'
A `float` number indicates the camera timestamp.

### 'magnetic time'
A `float` number indicates the sensor array timestamp.











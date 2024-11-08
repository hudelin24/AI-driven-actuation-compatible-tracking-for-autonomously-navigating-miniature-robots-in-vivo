# AI-driven-actuation-compatible-tracking-for-autonomously-navigating-miniature-robots-in-vivo

This is the implementation of our paper "AI-driven actuation-compatible tracking for autonomously navigating miniature robots in vivo", which is currently under review.

# Hardware requirements
We recommend running the code on a single RTX 4090 GPU.

# Installation
First, create a conda virtual environment and activate it:
```
conda create -n MagTrackTransformer python=3.9 -y
conda activate MagTrackTransformer
```
Then, install the following packages:

- [Pytorch with cuda](https://pytorch.org): `conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: ```pip install simplejson```
- einops: ```pip install einops```
- numpy: ```pip install numpy```
- tensorboard: ```pip install tensorboard```

Finally, build the MagTrackTransformer codebase by running:

```
git clone https://github.com/hudelin24/AI-driven-actuation-compatible-tracking-for-autonomously-navigating-miniature-robots-in-vivo
cd AI-driven-actuation-compatible-tracking-for-autonomously-navigating-miniature-robots-in-vivo
```
# Usage
## Data Preparation
Download our data via [MCT_pretrain](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186674_link_cuhk_edu_hk/ETzXjOHRdRRHrmgr_CimL04BO60uYoOhaof6tffcEZvlbA?e=2K9tVg), [MCT_finetune](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186674_link_cuhk_edu_hk/EZdvWbMEFGZOjkqtx2yJxRIBt8v-2DGbTgDTK9BgPM9sZg?e=01XX0w), [MCT_calib](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186674_link_cuhk_edu_hk/EdvXqDIXvDZOh1vaX37hPV4BldFXVGJuC9vjeHWnts884g?e=dS5RVl), [MTT](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186674_link_cuhk_edu_hk/EdFl4iB_PRhDmjAx4fG0CW8BBp1ivkXJEucrjdyc7Do9Mw?e=uQcSkS), or from Zenodo (release soon) to `Data` folder and uncompress. A desrpcetion about our data is provided in Data.md.


## MCT pre-training
Pre-train the MCT with default settings.
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_pretrain.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MCT_pretrain \
  OUTPUT_DIR MagTrackTransformer/results/calib_p 
```
The results will be saved at OUTPUT_DIR, i.e., `MagTrackTransformer/results/calib_p`.

## MCT fine-tuning
Fine-tune the pretrained MCT.
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_finetune.yaml \
  GPU_ENABLE True \
  TRAIN.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/pretrained_MCT/checkpoint_epoch_00140.pyth \
  DATA.PATH_TO_DATA_DIR Data/MCT_finetune/MWMR_S/calib_mtt_train_1 \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_S/calib_mtt_train_1 
  
```
We can use different datasets to finetune the pre-trained MCT (by changing DATA.PATH_TO_DATA_DIR), and save the results to corresponding folder (by changing OUTPUT_DIR). For example:
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_finetune.yaml \
  GPU_ENABLE True \
  TRAIN.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/pretrained_MCT/checkpoint_epoch_00140.pyth \
  DATA.PATH_TO_DATA_DIR Data/MCT_finetune/MWMR_L/calib_mtt_val_2 \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_L/calib_mtt_val_2 
```

## Data calibration (interference filtering) with fine-tuned MCTs
Remove the interference in the magnetic readouts using the corresponding fine-tuned MCT. For example, if we want to calibrate the interference in the data saved in `Data/MCT_calib/MWMR_S/calib_mtt_train_1`, we need to use the finetuned MCT saved at `MagTrackTransformer/trained_NNs/finetuned_MCT/MWMR_S/calib_mtt_train_1/checkpoint_epoch_00066.pyth`.

```
python MagTrackTransformer/tools/run_calib.py \
  --cfg MagTrackTransformer/configs/calib/MCT_calib.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MCT_calib/MWMR_S/calib_mtt_train_1 \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_S/calib_mtt_train_1 \
  TEST.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/finetuned_MCT/MWMR_S/calib_mtt_train_1/checkpoint_epoch_00066.pyth 
```
If we want to calibrate the interference in the data saved in `Data/MCT_calib/MWMR_L/calib_mtt_test_2`, we need to use the finetuned MCT saved at `MagTrackTransformer/trained_NNs/finetuned_MCT/MWMR_L/calib_mtt_test_2/checkpoint_epoch_00024.pyth`.
```
python MagTrackTransformer/tools/run_calib.py \
  --cfg MagTrackTransformer/configs/calib/MCT_calib.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MCT_calib/MWMR_L/calib_mtt_test_2 \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_L/calib_mtt_test_2 \
  TEST.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/finetuned_MCT/MWMR_L/calib_mtt_test_2/checkpoint_epoch_00024.pyth 
```
## MTT training
After data calibration (interference filtering), the estimated signals emitted by the miniature wireless magnetic robots can be obtained. The MTT is designed to implement spatiotemporal attention on these signals for localization. 

### MWMR_S
Train the MTT for MWMR_S with default settings by  
 
```
python MagTrackTransformer/tools/run_mtt.py \
  --cfg MagTrackTransformer/configs/track/MTT_train.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MTT/MWMR_S/ \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_S/tracking_divide_128_10_4_0.1 
```
Test the trained MTT on the testing set
```
python MagTrackTransformer/tools/run_mtt.py \
  --cfg MagTrackTransformer/configs/track/MTT_train.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MTT/MWMR_S/ \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_S/tracking_divide_128_10_4_0.1 \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/trained_MTT/MWMR_S/tracking_divide_128_10_4_0.1/checkpoint_epoch_00040.pyth 
```
### MWMR_L
Train the MTT for MWMR_L (depth=8 and drop_path_rate=0.2):
```
python MagTrackTransformer/tools/run_mtt.py \
  --cfg MagTrackTransformer/configs/track/MTT_train.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MTT/MWMR_L/ \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_L/tracking_divide_128_8_4_0.2 \
  MODEL_MTT.DEPTH 8 \
  MODEL_MTT.DROP_PATH_RATE 0.2 
```
Test the trained MTT on the testing set
```
python MagTrackTransformer/tools/run_mtt.py \
  --cfg MagTrackTransformer/configs/track/MTT_train.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MTT/MWMR_L/ \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_L/tracking_divide_128_8_4_0.2 \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/trained_MTT/MWMR_L/tracking_divide_128_8_4_0.2/checkpoint_epoch_00020.pyth \
  MODEL_MTT.DEPTH 8 \
  MODEL_MTT.DROP_PATH_RATE 0.2 
```




















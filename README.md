# AI-driven-actuation-compatible-tracking-for-autonomously-navigating-miniature-robots-in-vivo

This is the implementation of our paper "AI-driven actuation-compatible tracking for autonomously navigating miniature robots in vivo", which is currently under review.


# Installation
First, create a conda virtual environment and activate it:
```
conda create -n MagTrackTransformer python=3.9 -y
conda activate timesformer
```
Then, install the following packages:

- [Pytorch with cuda](https://pytorch.org): `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
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
Download our data from Zenodo (release soon) to Data folder and uncompress. A desrpcetion about our data is provided in Data.md.


## MCT pre-training
Pre-train the MCT with default settings.
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_pretrain.yaml \
  GPU_ENABLE True \
  DATA.PATH_TO_DATA_DIR Data/MCT_pretrain \
  OUTPUT_DIR MagTrackTransformer/results/calib_p \
```
The results will be saved at OUTPUT_DIR, i.e., MagTrackTransformer/results/calib_p.

## MCT fine-tuning
Fine-tune the pretrained MCT.
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_finetune.yaml \
  GPU_ENABLE True \
  TRAIN.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/pretrained_MCT/checkpoint_epoch_00140.pyth \
  DATA.PATH_TO_DATA_DIR Data/MCT_finetune/MWMR_S/calib_mtt_train_1 \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_S/calib_mtt_train_1 \
  
```
We can use different datasets to finetune the pre-trained MCT (by changing DATA.PATH_TO_DATA_DIR), and save the results to corresponding folder (by changing OUTPUT_DIR). For example:
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_finetune.yaml \
  GPU_ENABLE True \
  TRAIN.CHECKPOINT_FILE_PATH MagTrackTransformer/trained_NNs/pretrained_MCT/checkpoint_epoch_00140.pyth \
  DATA.PATH_TO_DATA_DIR Data/MCT_finetune/MWMR_L/calib_mtt_val_2 \
  OUTPUT_DIR MagTrackTransformer/results/MWMR_L/calib_mtt_val_2 \
```

## Data calibration (interference filtering) with fine-tuned MCTs
Remove the interference in the magnetic readouts using fine-tuned MCTs
























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

Then, build the MagTrackTransformer codebase by running:

```
git clone https://github.com/hudelin24/AI-driven-actuation-compatible-tracking-for-autonomously-navigating-miniature-robots-in-vivo
cd AI-driven-actuation-compatible-tracking-for-autonomously-navigating-miniature-robots-in-vivo
```
# Usage
## Data Preparation
Download our data from Zenodo (release soon) to Data folder and uncompress. A desrpcetion about our data is provided in Data.md.


## MCT pretraining
Pretrain the default MCT
```
python MagTrackTransformer/tools/run_mct.py \
  --cfg MagTrackTransformer/configs/calib/MCT_pretrain.yaml \
  GPU_ENABLE True \
  PATH_TO_DATA_DIR Data/MCT_pretrain \
  OUTPUT_DIR MagTrackTransformer/results/calib_p \
```
The results will be saved at MagTrackTransformer/results/calib_p.







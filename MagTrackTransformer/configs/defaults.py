from fvcore.common.config import CfgNode
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# -----------------------------------------------------------------------------
# Model MCT options
# -----------------------------------------------------------------------------
_C.MODEL_MCT = CfgNode()

# Magnetic map size [in_chans, H, W]
_C.MODEL_MCT.MAG_SIZE = [3, 4, 7]

#Number of calibration units 
_C.MODEL_MCT.NUM_CALIB = 12

#Embedding dimension
_C.MODEL_MCT.EMBED_DIM = 128

#Number of transformer blocks
_C.MODEL_MCT.DEPTH = 6

#Number of heads
_C.MODEL_MCT.NUM_HEADS = 8

#Inflation ratio for mlp hiddern dimension
_C.MODEL_MCT.MLP_RATIO = 2

#Biase for linear qkv
_C.MODEL_MCT.QKV_BIAS = True

#Scale for qkv
_C.MODEL_MCT.QKV_SCALE = None

#Tranformer drop rate
_C.MODEL_MCT.DROP_RATE = 0.0

#Attention drop rate = 0
_C.MODEL_MCT.ATTN_DROP_RATE = 0.0

#Drop path rate 
_C.MODEL_MCT.DROP_PATH_RATE = 0.1

# Loss function.
_C.MODEL_MCT.LOSS_FUNC = "mse"

# -----------------------------------------------------------------------------
# Model MTT options
# -----------------------------------------------------------------------------
_C.MODEL_MTT = CfgNode()

# Magnetic map size [in_chans, T, H, W]
_C.MODEL_MTT.MAG_SIZE = [3, 256, 4, 7]

#Magnetic map patch size (T, H, W) 
_C.MODEL_MTT.PATCH_SIZE = [8, 2, 3]

#Magnetic map stride size (T, H, W) 
_C.MODEL_MTT.STRIDE_SIZE = [8, 2, 2]

#Tracking output dim 
_C.MODEL_MTT.OUT_DIM = 3

#Embedding dimension
_C.MODEL_MTT.EMBED_DIM = 128

#Number of transformer blocks
_C.MODEL_MTT.DEPTH = 6

#Number of heads
_C.MODEL_MTT.NUM_HEADS = 8

#Inflation ratio for mlp hiddern dimension
_C.MODEL_MTT.MLP_RATIO = 2

#Biase for linear qkv
_C.MODEL_MTT.QKV_BIAS = True

#Scale for qkv
_C.MODEL_MTT.QKV_SCALE = None

#Tranformer drop rate
_C.MODEL_MTT.DROP_RATE = 0.0

#Attention drop rate = 0
_C.MODEL_MTT.ATTN_DROP_RATE = 0.0

#Drop path rate 
_C.MODEL_MTT.DROP_PATH_RATE = 0.1

#Attention type 'divided_space_time' or 'joint_space_time'
_C.MODEL_MTT.ATTN_TYPE = "joint_space_time"

# Loss function.
_C.MODEL_MTT.LOSS_FUNC = "mse"


# -----------------------------------------------------------------------------
# Model MOT options
# -----------------------------------------------------------------------------
_C.MODEL_MOT = CfgNode()

# Magnetic map size [in_chans, H, W]
_C.MODEL_MOT.MAG_SIZE = [3, 4, 7]

#AUXILIARY dimension (estimated position)
_C.MODEL_MOT.AUXILIARY_DIM = 3

#Embedding dimension
_C.MODEL_MOT.EMBED_DIM = 128

#Orientation dimension
_C.MODEL_MOT.ORIENT_DIM = 3

#Number of transformer blocks
_C.MODEL_MOT.DEPTH = 6

#Number of heads
_C.MODEL_MOT.NUM_HEADS = 8

#Inflation ratio for mlp hiddern dimension
_C.MODEL_MOT.MLP_RATIO = 2

#Biase for linear qkv
_C.MODEL_MOT.QKV_BIAS = True

#Scale for qkv
_C.MODEL_MOT.QKV_SCALE = None

#Tranformer drop rate
_C.MODEL_MOT.DROP_RATE = 0.0

#Attention drop rate = 0
_C.MODEL_MOT.ATTN_DROP_RATE = 0.0

#Drop path rate 
_C.MODEL_MOT.DROP_PATH_RATE = 0.1

# Loss function
_C.MODEL_MOT.LOSS_FUNC = "mse"

# AUXILIARY_ENABLE
_C.MODEL_MOT.AUXILIARY_ENABLE = True

# MAG Noise ENABLE
_C.MODEL_MOT.MAG_NOISE_ENABLE = True

# MAG Noise DIR
_C.MODEL_MOT.MAG_NOISE_DIR = "Data_final/MOT/MWMR_S/mag_noise_distribution.pyth"

# AUXILIARY Noise ENABLE
_C.MODEL_MOT.AUXILIARY_NOISE_ENABLE = True

# AUXILIARY Noise DIR
_C.MODEL_MOT.AUXILIARY_NOISE_DIR = "Data_final/MOT/MWMR_S/auxiliary_noise_distribution.pyth"


# -----------------------------------------------------------------------------
# Model MAT options
# -----------------------------------------------------------------------------
_C.MODEL_MAT = CfgNode()

# Magnetic map size [in_chans, H, W]
_C.MODEL_MAT.MAG_SIZE = [3, 4, 7]

#Embedding dimension
_C.MODEL_MAT.EMBED_DIM = 128

#Orientation dimension
_C.MODEL_MAT.ORIENT_DIM = 3

#Position dimension
_C.MODEL_MAT.POS_DIM = 3

#Number of transformer blocks
_C.MODEL_MAT.DEPTH = 6

#Number of heads
_C.MODEL_MAT.NUM_HEADS = 8

#Inflation ratio for mlp hiddern dimension
_C.MODEL_MAT.MLP_RATIO = 2

#Biase for linear qkv
_C.MODEL_MAT.QKV_BIAS = True

#Scale for qkv
_C.MODEL_MAT.QKV_SCALE = None

#Tranformer drop rate
_C.MODEL_MAT.DROP_RATE = 0.0

#Attention drop rate = 0
_C.MODEL_MAT.ATTN_DROP_RATE = 0.0

#Drop path rate 
_C.MODEL_MAT.DROP_PATH_RATE = 0.1

# Loss function
_C.MODEL_MAT.LOSS_FUNC_POS = "mse"
_C.MODEL_MAT.LOSS_FUNC_ORI = "mse"
_C.MODEL_MAT.LOSS_FUNC_WEIGHT = 1.0


# -----------------------------------------------------------------------------
# Model IMAT options
# -----------------------------------------------------------------------------
_C.MODEL_IMAT = CfgNode()

# number of magnetic readouts
_C.MODEL_IMAT.NUM_MAG = 40

# #Embedding dimension
_C.MODEL_IMAT.EMBED_DIM = 128

#Orientation dimension
_C.MODEL_IMAT.ORIENT_DIM = 3

#Position dimension
_C.MODEL_IMAT.POS_DIM = 3

#Output dimension
_C.MODEL_IMAT.OUT_DIM = 3

#Number of transformer blocks
_C.MODEL_IMAT.DEPTH = 6

#Number of heads
_C.MODEL_IMAT.NUM_HEADS = 8

#Inflation ratio for mlp hiddern dimension
_C.MODEL_IMAT.MLP_RATIO = 2

#Biase for linear qkv
_C.MODEL_IMAT.QKV_BIAS = True

#Scale for qkv
_C.MODEL_IMAT.QKV_SCALE = None

#Tranformer drop rate
_C.MODEL_IMAT.DROP_RATE = 0.0

#Attention drop rate = 0
_C.MODEL_IMAT.ATTN_DROP_RATE = 0.0

#Drop path rate 
_C.MODEL_IMAT.DROP_PATH_RATE = 0.1

# Loss function
_C.MODEL_IMAT.LOSS_FUNC = "mse"

# -----------------------------------------------------------------------------
# Model MDT options
# -----------------------------------------------------------------------------
_C.MODEL_MDT = CfgNode()

# size of TUSs readouts
_C.MODEL_MDT.TSUS_SIZE = (3,2,4,7)

# size of CUSs readouts
_C.MODEL_MDT.CSUS_SIZE = (3,2,12)

#AUXILIARY dimension (estimated position)
_C.MODEL_MDT.AUXILIARY_DIM = 3

# #Embedding dimension
_C.MODEL_MDT.EMBED_DIM = 128

#Number of transformer blocks
_C.MODEL_MDT.DEPTH = 6

#Number of heads
_C.MODEL_MDT.NUM_HEADS = 8

#Inflation ratio for mlp hiddern dimension
_C.MODEL_MDT.MLP_RATIO = 2

#Biase for linear qkv
_C.MODEL_MDT.QKV_BIAS = True

#Scale for qkv
_C.MODEL_MDT.QKV_SCALE = None

#Tranformer drop rate
_C.MODEL_MDT.DROP_RATE = 0.0

#Attention drop rate = 0
_C.MODEL_MDT.ATTN_DROP_RATE = 0.0

#Drop path rate 
_C.MODEL_MDT.DROP_PATH_RATE = 0.1

# Loss function
_C.MODEL_MDT.LOSS_FUNC = "mse"

# Noise path
_C.MODEL_MDT.POST_CALIB_NOISE_DIR = ""

# Noise enable
_C.MODEL_MDT.POST_CALIB_ENABLE = True

# Noise free ratio
_C.MODEL_MDT.NOISE_FREE_RATIO = 0.1

# AUXILIARY_ENABLE
_C.MODEL_MDT.AUXILIARY_ENABLE = False

# Trained MDT PATH 
_C.MODEL_MDT.CHECKPOINT_FILE_PATH = ""


# -----------------------------------------------------------------------------
# Model MLP options
# -----------------------------------------------------------------------------
_C.MODEL_MLP = CfgNode()

#Input dimension
_C.MODEL_MLP.IN_DIM = 84

#Hidden layer dimension
_C.MODEL_MLP.HIDDEN_DIM = 100

#Output dimension
_C.MODEL_MLP.OUT_DIM = 3

#Number of hidden layers
_C.MODEL_MLP.NUM_HIDDEN_LAYER = 3


# -----------------------------------------------------------------------------
# Model PoseNet options
# -----------------------------------------------------------------------------
_C.MODEL_PSN = CfgNode()

#Output dimension
_C.MODEL_PSN.OUT_DIM = 3

# -----------------------------------------------------------------------------
# Model PoseNet3D options
# -----------------------------------------------------------------------------
_C.MODEL_PSN3D = CfgNode()

#Output dimension
_C.MODEL_PSN3D.OUT_DIM = 3


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adamw"

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# End learning rate.
_C.SOLVER.COSINE_END_LR = 0.001

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "calib"

# Batch size.
_C.TRAIN.BATCH_SIZE = 64

# Checkpoint types include `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

#
_C.TRAIN.FINETUNE = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "calib"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 64

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "/Users/dlh/Desktop/latex/CUHK/py/Processed_data/pub1/calib/"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = True

# Default to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""



# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

#If True, implement on gpu
_C.GPU_ENABLE = True

#ID of the gpu to use
_C.GPU_ID = None

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Output basedir.
_C.OUTPUT_DIR = "/Magformer/results"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Path to the checkpoint to load the pretrained mct parameters.
_C.MCT_CHECKPOINT_FILE_PATH = ""

# Model name
_C.MODEL_NAME = "mct_base"

# Data augumentation
_C.DATA_AUGUMENTATION = False



def _assert_and_infer_cfg(cfg):
 
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch"]
 
    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch"]

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())

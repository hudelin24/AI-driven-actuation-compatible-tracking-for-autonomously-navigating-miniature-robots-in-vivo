import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from tools.test_mdt import test
from tools.train_mdt import train
from utlis.parser import load_config, parse_args


def get_func(cfg):
    train_func = train
    test_func = test
    return train_func, test_func

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    train, test = get_func(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        cfg.TEST.CHECKPOINT_FILE_PATH = train(cfg)

    # Perform testing.
    if cfg.TEST.ENABLE:
        test(cfg)

if __name__ == "__main__":
    main()


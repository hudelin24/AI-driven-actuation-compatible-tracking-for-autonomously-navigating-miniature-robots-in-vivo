import torch
import datetime
import numpy as np
import os
import utlis.logging as logging
from fvcore.common.timer import Timer
from collections import deque
import utlis.misc as misc
from fvcore.common.file_io import PathManager

logger = logging.get_logger(__name__)

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_l1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_euclidean_err = ScalarMeter(cfg.LOG_PERIOD)

        self.accum_l1_err = 0
        self.accum_euclidean_err = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_l1_err.reset()
        self.mb_euclidean_err.reset()
        self.accum_l1_err = 0
        self.accum_euclidean_err = 0
        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0


    def update_stats(self, l1_err, euclidean_err, loss, lr, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            l1_err (float): l1 error.
            euclidean_err (float): euclidean error.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.mb_l1_err.add_value(l1_err)
        self.mb_euclidean_err.add_value(euclidean_err)
        # Aggregate stats
        self.accum_l1_err += l1_err * mb_size
        self.accum_euclidean_err += euclidean_err * mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["l1_err"] = self.mb_l1_err.get_win_median()
        stats["euclidean_err"] = self.mb_euclidean_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        l1_err = self.accum_l1_err / self.num_samples
        euclidean_err = self.accum_euclidean_err / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats["l1_err"] = l1_err
        stats["euclidean_err"] = euclidean_err
        stats["loss"] = avg_loss
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_l1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_euclidean_err = ScalarMeter(cfg.LOG_PERIOD)

        self.accum_l1_err = 0
        self.accum_euclidean_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.mb_l1_err.reset()
        self.mb_euclidean_err.reset()
        self.accum_l1_err = 0
        self.accum_euclidean_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def update_stats(self, l1_err, euclidean_err, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            l1_err (float): l1 error.
            euclidean_err (float): euclidean error.
            mb_size (int): mini batch size.
        """
        self.mb_l1_err.add_value(l1_err)
        self.mb_euclidean_err.add_value(euclidean_err)
        self.accum_l1_err += l1_err * mb_size
        self.accum_euclidean_err += euclidean_err * mb_size
        self.num_samples += mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["l1_err"] = self.mb_l1_err.get_win_median()
        stats["euclidean_err"] = self.mb_euclidean_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        l1_err = self.accum_l1_err / self.num_samples
        euclidean_err = self.accum_euclidean_err / self.num_samples

        stats["l1_err"] = l1_err
        stats["euclidean_err"] = euclidean_err

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

        return l1_err
    
class TestMeter(object):
    """
    Measures testing stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        # Current minibatch errors (smoothed over a window).
        self.mb_l1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_euclidean_err = ScalarMeter(cfg.LOG_PERIOD)
        self.accum_l1_err = 0
        self.accum_euclidean_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD
        
        
    def reset(self):
        """
        Reset the Meter.
        """
        self.mb_l1_err.reset()
        self.mb_euclidean_err.reset()
        self.accum_l1_err = 0
        self.accum_euclidean_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def update_stats(self, l1_err, euclidean_err, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            l1_err (float): l1 error.
            euclidean_err (float): euclidean error.
            mb_size (int): mini batch size.
        """
        self.mb_l1_err.add_value(l1_err)
        self.mb_euclidean_err.add_value(euclidean_err)
        self.accum_l1_err += l1_err * mb_size
        self.accum_euclidean_err += euclidean_err * mb_size
        self.num_samples += mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size


    def update_predictions(self, preds, labels=None):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor or None): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        if labels is not None:
            self.all_labels.append(labels)

    def save_predictions(self):
        """
        Save predictions and labels.
        """
        path_to_save = os.path.join(self.output_dir, "preds.pyth")
        file_to_save = {"preds": [pred.clone().detach().cpu() for pred in self.all_preds]}
        if len(self.all_labels) == len(self.all_preds):
            file_to_save["labels"] = [label.clone().detach().cpu() for label in self.all_labels]
        else:
            file_to_save["labels"] = None

        
        torch.save(file_to_save, path_to_save)
        logger.info("Predictions save to {}".format(path_to_save))

    def log_iter_stats(self, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return

        stats = {
            "_type": "test_iter",
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["l1_err"] = self.mb_l1_err.get_win_median()
        stats["euclidean_err"] = self.mb_euclidean_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()
        logging.log_json_stats(stats)

    def log_stats(self):
        """
        Log the stats of the current epoch.
        """
        stats = {
            "_type": "test",
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        l1_err = self.accum_l1_err / self.num_samples
        euclidean_err = self.accum_euclidean_err / self.num_samples

        stats["l1_err"] = l1_err
        stats["euclidean_err"] = euclidean_err

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples

        logging.log_json_stats(stats)


class TrackingRawDataMeter(object):
    """
    Process and record tracking data.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.all_tracking_mag_map = []
        self.all_path_to_tracking = []

    def reset(self):
        """
        Reset the Meter.
        """
        self.all_tracking_mag_map = []
        self.all_path_to_tracking = []

    def update_data(self, tracking_mag_map, path_to_tracking):
        """
        Update the Meter.
        """
        self.all_tracking_mag_map.append(tracking_mag_map)          #[bsz, in_chans * 2, 1, H, W]
        self.all_path_to_tracking.append(path_to_tracking)          #[bsz]

    def save_data(self):
        """
        Save data to tracking files.
        """
        for i in range(len(self.all_tracking_mag_map)):
            for j in range(self.all_tracking_mag_map[i].shape[0]):
                with PathManager.open(self.all_path_to_tracking[i][j], "rb") as f:
                    dic_tracking = torch.load(f, map_location="cpu")
                    #print(list(dic_tracking.keys()))
                    dic_tracking["calibrated magetic map"] = self.all_tracking_mag_map[i][j,:,:,:,:].unsqueeze(0).clone().detach().cpu()
                
                with PathManager.open(self.all_path_to_tracking[i][j], "wb") as f:
                    torch.save(dic_tracking, f)


class TrackingNoiseRawDataMeter(object):
    """
    Process and record tracking data.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.all_tracking_mag_map = []
        self.all_path_to_tracking = []

    def reset(self):
        """
        Reset the Meter.
        """
        self.all_tracking_mag_map = []
        self.all_path_to_tracking = []

    def update_data(self, tracking_mag_map, path_to_tracking):
        """
        Update the Meter.
        """
        self.all_tracking_mag_map.append(tracking_mag_map)          #[bsz, in_chans * 2, 1, H, W]
        self.all_path_to_tracking.append(path_to_tracking)          #[bsz]

    def save_data(self):
        """
        Save data to tracking files.
        """
        for i in range(len(self.all_tracking_mag_map)):
            for j in range(self.all_tracking_mag_map[i].shape[0]):
                with PathManager.open(self.all_path_to_tracking[i][j], "rb") as f:
                    dic_tracking = torch.load(f, map_location="cpu")
                    #print(list(dic_tracking.keys()))
                    dic_tracking["calibrated denoised magnetic map"] = self.all_tracking_mag_map[i][j,:,:,:,:].unsqueeze(0).clone().detach().cpu()
                
                with PathManager.open(self.all_path_to_tracking[i][j], "wb") as f:
                    #print(self.all_path_to_tracking[i][j])
                    torch.save(dic_tracking, f)


class TrainMeter_orient(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_angular_err = ScalarMeter(cfg.LOG_PERIOD)
        self.accum_angular_err = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_angular_err.reset()
        self.accum_angular_err = 0
        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0


    def update_stats(self, angular_err, loss, lr, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            angular_err (float): angular error.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.mb_angular_err.add_value(angular_err)
        # Aggregate stats
        self.accum_angular_err += angular_err * mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["angular_err"] = self.mb_angular_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        angular_err = self.accum_angular_err / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats["angular_err"] = angular_err
        stats["loss"] = avg_loss
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)


class ValMeter_orient(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_angular_err = ScalarMeter(cfg.LOG_PERIOD)
        self.accum_angular_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.mb_angular_err.reset()
        self.accum_angular_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def update_stats(self, angular_err, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            angular_err (float): angular error.
            mb_size (int): mini batch size.
        """
        self.mb_angular_err.add_value(angular_err)
        self.accum_angular_err += angular_err * mb_size
        self.num_samples += mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["angular_err"] = self.mb_angular_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        angular_err = self.accum_angular_err / self.num_samples

        stats["angular_err"] = angular_err

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

        return angular_err


class TestMeter_orient(object):
    """
    Measures testing stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        # Current minibatch errors (smoothed over a window).
        self.mb_angular_err = ScalarMeter(cfg.LOG_PERIOD)
        self.accum_angular_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD
        
        
    def reset(self):
        """
        Reset the Meter.
        """
        self.mb_angular_err.reset()
        self.accum_angular_err = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def update_stats(self, angular_err, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            angular_err (float): angular error.
            mb_size (int): mini batch size.
        """
        self.mb_angular_err.add_value(angular_err)
        self.accum_angular_err += angular_err * mb_size
        self.num_samples += mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size


    def update_predictions(self, preds, labels=None):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor or None): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        if labels is not None:
            self.all_labels.append(labels)

    def save_predictions(self):
        """
        Save predictions and labels.
        """
        path_to_save = os.path.join(self.output_dir, "preds_ori.pyth")
        file_to_save = {"preds": [pred.clone().detach().cpu() for pred in self.all_preds]}
        if len(self.all_labels) == len(self.all_preds):
            file_to_save["labels"] = [label.clone().detach().cpu() for label in self.all_labels]
        else:
            file_to_save["labels"] = None

        
        torch.save(file_to_save, path_to_save)
        logger.info("Predictions save to {}".format(path_to_save))

    def log_iter_stats(self, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return

        stats = {
            "_type": "test_iter",
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["angular_err"] = self.mb_angular_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()
        logging.log_json_stats(stats)

    def log_stats(self):
        """
        Log the stats of the current epoch.
        """
        stats = {
            "_type": "test",
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        angular_err = self.accum_angular_err / self.num_samples
        stats["angular_err"] = angular_err

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples

        logging.log_json_stats(stats)


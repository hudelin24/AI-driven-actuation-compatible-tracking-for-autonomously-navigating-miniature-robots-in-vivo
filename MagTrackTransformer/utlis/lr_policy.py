import math


def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
    return lr

def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    assert cfg.SOLVER.COSINE_END_LR < cfg.SOLVER.BASE_LR
    return (
        cfg.SOLVER.COSINE_END_LR
        + (cfg.SOLVER.BASE_LR - cfg.SOLVER.COSINE_END_LR)
        * (math.cos(math.pi * cur_epoch / cfg.SOLVER.MAX_EPOCH) + 1.0)
        * 0.5
    )

def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]

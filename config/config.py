#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.SESSION = 'TrainTest'
        self._C.MODEL.LL = 'UNet'
        self._C.MODEL.ENHANCE = 'None'
        self._C.MODEL.FILM = 'None'
        self._C.MODEL.INPUT = 'input'
        self._C.MODEL.TARGET = 'target'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.SEED = 3407
        self._C.OPTIM.NUM_EPOCHS = 90
        self._C.OPTIM.NEPOCH_DECAY = [50]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5
        self._C.OPTIM.WANDB = False
                # ==== Muon 相关配置（新增） ====
        self._C.OPTIM.USE_MUON = False         # 是否启用 Muon
        self._C.OPTIM.WEIGHT_DECAY = 0.0       # 通用权重衰减

        # 学习率策略：优先使用 MUON_LR，如果没填则用 LR_INITIAL * MUON_LR_MULT
        self._C.OPTIM.MUON_LR = None           # 显式指定 Muon lr（可留 None）
        self._C.OPTIM.MUON_LR_MULT = 5         # Muon 学习率放大倍数（典型=5）

        # 不再用“仅2D权重”这个开关；改为只对卷积核(4D)用 Muon
        # 保留字段但置 False（或直接删除这行都行）
        self._C.OPTIM.MUON_ONLY_2D = False


        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.WEIGHT = ''
        self._C.TRAINING.TRAIN_DIR = 'dataset'
        self._C.TRAINING.VAL_DIR = 'dataset'
        self._C.TRAINING.SAVE_DIR = 'checkpoints/Restormer'
        self._C.TRAINING.PS_W = 256
        self._C.TRAINING.PS_H = 256
        self._C.TRAINING.LOG_FILE = 'trainlog.txt'


        self._C.TESTING = CN()
        self._C.TESTING.INPUT = 'input'
        self._C.TESTING.TARGET = 'target'
        self._C.TESTING.VAL_DIR = 'dataset'
        self._C.TESTING.WEIGHT = None
        self._C.TESTING.SAVE_IMAGES = False
        self._C.TESTING.PS_W = 256
        self._C.TESTING.PS_H = 256
        self._C.TESTING.RESULT_DIR = 'checkpoints/Restormer'
        self._C.TESTING.LOG_FILE = 'testlog.txt'

        # === LOG 配置 ===
        self._C.LOG = CN()
        self._C.LOG.LOG_DIR = './log_dir/'


        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()

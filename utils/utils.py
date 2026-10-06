import os
import random
from collections import OrderedDict

import numpy as np
import torch


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def save_checkpoint(state, epoch, outdir):
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#     checkpoint_file = os.path.join(outdir, 'epoch_' + str(epoch) + '.pth')
#     torch.save(state, checkpoint_file)

def save_checkpoint(state, epoch, outdir):
    """
    自动判断 outdir 是文件夹还是完整文件路径：
      - 如果以 .pth 结尾 → 直接保存为该文件
      - 否则 → 认为是文件夹，自动命名为 epoch_{epoch}.pth
    """
    # 判断是否为完整路径（例如 last_epoch_41.pth）
    if outdir.endswith(".pth"):
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        torch.save(state, outdir)
    else:
        os.makedirs(outdir, exist_ok=True)
        checkpoint_file = os.path.join(outdir, f'epoch_{epoch}.pth')
        torch.save(state, checkpoint_file)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'll_layer_module' not in key:
            key = key.replace('ll_layer', 'll_layer_module')
        if 'enhance_module' not in key:
            key = key.replace('enhance', 'enhance_module')
        if key.startswith('module'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)

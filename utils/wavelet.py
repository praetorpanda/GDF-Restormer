import torch
import pywt
import numpy as np
import torch.nn.functional as F


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def to_tensor(ndarray, device=None):
    tensor = torch.from_numpy(ndarray).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def dwt2_tensor(img_tensor, wavelet='haar', mode='symmetric', return_all=False):
    """
    执行单层二维离散小波变换。

    参数：
    - img_tensor: (B, C, H, W) Tensor，值域 [0, 1]
    - wavelet: 小波基，如 'haar', 'db1', 'sym2', 'coif1' 等
    - mode: 边界填充方式，参考 pywt 文档
    - return_all: 如果为 False，仅返回低频 LL；否则返回 (LL, [LH, HL, HH])
    """
    B, C, H, W = img_tensor.shape
    results_LL, results_LH, results_HL, results_HH = [], [], [], []

    for b in range(B):
        for c in range(C):
            img = img_tensor[b, c]
            coeffs2 = pywt.dwt2(to_numpy(img), wavelet=wavelet, mode=mode)
            LL, (LH, HL, HH) = coeffs2
            results_LL.append(LL)
            if return_all:
                results_LH.append(LH)
                results_HL.append(HL)
                results_HH.append(HH)

    LL_tensor = to_tensor(np.stack(results_LL)).reshape(B, C, *LL.shape)
    if not return_all:
        return LL_tensor
    else:
        LH_tensor = to_tensor(np.stack(results_LH)).reshape(B, C, *LH.shape)
        HL_tensor = to_tensor(np.stack(results_HL)).reshape(B, C, *HL.shape)
        HH_tensor = to_tensor(np.stack(results_HH)).reshape(B, C, *HH.shape)
        return LL_tensor, LH_tensor, HL_tensor, HH_tensor


def idwt2_tensor(LL, LH=None, HL=None, HH=None, wavelet='haar', mode='symmetric'):
    """
    单层逆小波变换（用于图像重构）

    - 输入：
        LL, LH, HL, HH: 形状为 (B, C, H', W') 的张量
    - 输出：
        重建图像：形状 (B, C, H, W)
    """
    B, C, _, _ = LL.shape
    recon = []
    for b in range(B):
        for c in range(C):
            ll = to_numpy(LL[b, c])
            if LH is not None and HL is not None and HH is not None:
                lh = to_numpy(LH[b, c])
                hl = to_numpy(HL[b, c])
                hh = to_numpy(HH[b, c])
                coeffs = (ll, (lh, hl, hh))
            else:
                # 全 0 高频时，近似重构
                shape = ll.shape
                coeffs = (ll, (np.zeros(shape), np.zeros(shape), np.zeros(shape)))
            rec = pywt.idwt2(coeffs, wavelet=wavelet, mode=mode)
            recon.append(rec)

    recon_tensor = to_tensor(np.stack(recon)).reshape(B, C, *rec.shape)
    return recon_tensor


def wavedec_tensor(img_tensor, wavelet='haar', level=2, mode='symmetric'):
    """
    多层小波分解 (二维)
    返回：
        [LL_n, (LH_n, HL_n, HH_n), ..., (LH_1, HL_1, HH_1)]
    """
    B, C, H, W = img_tensor.shape
    all_coeffs = []

    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec2(to_numpy(img_tensor[b, c]), wavelet=wavelet, level=level, mode=mode)
            all_coeffs.append(coeffs)

    return all_coeffs  # 注意：这不是张量，是 pywt 原生结构


def waverec_tensor(coeffs_list, wavelet='haar', mode='symmetric'):
    """
    多层逆小波重构（用于 wavedec_tensor 的输出）

    参数：
        coeffs_list: 每张图的 pywt.wavedec2 输出列表
    返回：
        重建图像 Tensor: (B, C, H, W)
    """
    BxC = len(coeffs_list)
    rec_list = []
    for coeffs in coeffs_list:
        img = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)
        rec_list.append(img)

    recon_tensor = to_tensor(np.stack(rec_list)).reshape(BxC, 1, *img.shape)
    return recon_tensor


def list_available_wavelets(family=None):
    """
    打印可用小波基名称
    """
    if family is None:
        return pywt.wavelist()
    else:
        return pywt.wavelist(family)

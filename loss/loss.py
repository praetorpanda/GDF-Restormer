import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision.models.vgg import vgg16, VGG16_Weights

class SSIMLoss(nn.Module):
    """结构相似性损失：1 - SSIM，带 clamp 以避免负值"""
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        ssim_val = ssim(pred, target, data_range=1.0)
        ssim_val = torch.clamp(ssim_val, min=0.0, max=1.0)
        return 1.0 - ssim_val




class Perceptual(torch.nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            module.to(x.device)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))
        return sum(loss) / len(loss)


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, inp, tar):
        lab_inp, lab_tar = rgb_to_lab(inp), rgb_to_lab(tar)
        l1, a1, b1 = torch.moveaxis(lab_inp, 1, 0)[:3]
        l2, a2, b2 = torch.moveaxis(lab_tar, 1, 0)[:3]
        return torch.mean(torch.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))


if __name__ == '__main__':
    tensor1 = torch.randn(1, 3, 360, 540)
    tensor2 = torch.randn(1, 3, 360, 540)
    loss = ColorLoss()
    l = loss(tensor1, tensor2)
    im_orig = tensor1.squeeze(0).permute(1, 2, 0).numpy()
    im_edit = tensor2.squeeze(0).permute(1, 2, 0).numpy()
    from skimage import color

    lab_orig = color.rgb2lab(im_orig)
    lab_edit = color.rgb2lab(im_edit)
    de_diff = color.deltaE_cie76(lab_orig, lab_edit)
    print(np.mean(de_diff))
    print(l)



# =========================
# 追加的鲁棒损失与组合器
# =========================


# ---- 工具：掩膜与边缘 ----
def make_valid_mask(inp, tar, black_thr=0.03, bright_thr=0.98):
    """
    构造简单掩膜，忽略黑边/过曝高光区域；值域建议在[0,1]
    返回: [B,1,H,W]
    """
    device, dtype = inp.device, inp.dtype
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=dtype).view(1, 3, 1, 1)
    g_inp = (inp * w).sum(1, keepdim=True)
    g_tar = (tar * w).sum(1, keepdim=True)
    valid = (g_inp > black_thr) & (g_tar > black_thr) & (g_inp < bright_thr) & (g_tar < bright_thr)
    return valid.float()

def sobel_edges(x):
    """Sobel 幅度图 [B,C,H,W]"""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

# ---- 基础鲁棒损失 ----
class CharbonnierLoss(nn.Module):
    """比 L1/L2 更抗 outlier 的像素损失"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask: torch.Tensor | None = None):
        diff = pred - target
        if mask is not None:
            diff = diff * mask
            denom = (mask.sum() / pred.shape[1]).clamp_min(1.0)  # 按通道平均
        else:
            denom = torch.tensor(pred.numel() / pred.shape[1], device=pred.device)
        return torch.sqrt(diff * diff + self.eps * self.eps).sum() / denom

class EdgeLoss(nn.Module):
    """Sobel 边缘一致性：在结构细节上更稳"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask: torch.Tensor | None = None):
        e_pred, e_tar = sobel_edges(pred), sobel_edges(target)
        diff = torch.abs(e_pred - e_tar)
        if mask is not None:
            diff = diff * mask
            denom = (mask.sum() / pred.shape[1]).clamp_min(1.0)
        else:
            denom = torch.tensor(pred.numel() / pred.shape[1], device=pred.device)
        return diff.sum() / denom

class TVLoss(nn.Module):
    """总变分：抑制噪点和伪纹理"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        loss = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        return self.weight * loss

# ---- LPIPS（可选；lazy initialization）----
# Importing loss.py must not construct LPIPS/VGG or load pretrained weights.
try:
    import lpips
    _has_lpips = True
except Exception:
    lpips = None
    _has_lpips = False

class LPIPSLoss(nn.Module):
    """LPIPS 感知损失（仅在实例化本类时加载 LPIPS/VGG）。"""
    def __init__(self):
        super().__init__()
        if not _has_lpips:
            raise ImportError("lpips not installed. `pip install lpips`")
        self.lp = lpips.LPIPS(net='vgg')
        self.lp.requires_grad_(False)

    def forward(self, pred, target):
        x = (pred.clamp(0,1) * 2 - 1).to(torch.float32)
        y = (target.clamp(0,1) * 2 - 1).to(torch.float32)
        return self.lp(x, y).mean()




# ---- 组合器：一处配置，统一调用 ----
class CombinedLoss(nn.Module):
    """
    weights: dict，键可包含：
      {'charb','ssim','edge','percep','lpips','tv','color'}
    use_mask: 是否启用掩膜（忽略黑边/高光）
    """
    def __init__(self, weights: dict, use_mask: bool = True, tv_weight: float | None = None):
        super().__init__()
        self.w = {
            'charb': float(weights.get('charb', 0.0)),
            'ssim':  float(weights.get('ssim', 0.0)),
            'edge':  float(weights.get('edge', 0.0)),
            'percep':float(weights.get('percep', 0.0)),
            'lpips': float(weights.get('lpips', 0.0)),
            'tv':    float(weights.get('tv', 0.0)),
            'color': float(weights.get('color', 0.0)),
        }
        self.use_mask = use_mask

        self.charb = CharbonnierLoss()
        self.edge  = EdgeLoss()
        self.tv    = TVLoss(weight=tv_weight if tv_weight is not None else 1.0)
        self.ssim_ = SSIMLoss()
        self.perc  = Perceptual()
        self.color = ColorLoss()
        self.lpips = LPIPSLoss() if _has_lpips and self.w['lpips'] > 0 else None

    def forward(self, pred, target, inp_for_mask=None):
        device = pred.device

        # 确保所有子模块在 pred 相同 device 上
        self.charb = self.charb.to(device)
        self.edge = self.edge.to(device)
        self.tv = self.tv.to(device)
        self.ssim_ = self.ssim_.to(device)
        self.perc = self.perc.to(device)
        self.color = self.color.to(device)
        if self.lpips is not None:
            self.lpips = self.lpips.to(device)

        mask = None
        if self.use_mask:
            base = inp_for_mask if inp_for_mask is not None else target
            try:
                mask = make_valid_mask(base, target).repeat(1, pred.shape[1], 1, 1)
            except Exception:
                mask = None

        pred = pred.clamp(0,1)
        target = target.clamp(0,1)

        parts = {}

        if self.w['charb'] > 0:
            parts['charb'] = self.charb(pred, target, mask=mask) * self.w['charb']
        if self.w['ssim'] > 0:
            parts['ssim']  = self.ssim_(pred, target) * self.w['ssim']
        if self.w['edge'] > 0:
            parts['edge']  = self.edge(pred, target, mask=mask) * self.w['edge']
        if self.w['percep'] > 0:
            parts['percep']= self.perc(pred, target) * self.w['percep']
        if self.lpips is not None:
            parts['lpips'] = self.lpips(pred, target) * self.w['lpips']
        if self.w['tv'] > 0:
            parts['tv']    = self.tv(pred) * self.w['tv']
        if self.w['color'] > 0:
            parts['color'] = self.color(pred, target) * self.w['color']

        total = sum(parts.values()) if len(parts) else torch.tensor(0.0, device=pred.device)
        return total, parts

# ---- 一个便捷的配置预设（可选）----
def build_combined_loss(preset: str = "real_robust"):
    """
    - real_robust：真实配对（有错配/高光），更鲁棒 + 感知 + 边缘
    - synth_precise：合成完美对齐，像素项更重
    """
    if preset == "real_robust":
        w = dict(charb=0.45, ssim=0.25, edge=0.15, percep=0.1, lpips=0.05, tv=0.05, color=0.00)
        # w = dict(charb=0.45, ssim=0.25, edge=0.15, percep=0.1, lpips=0.05, tv=0.0, color=0.00)
        return CombinedLoss(weights=w, use_mask=True)
    elif preset == "synth_precise":
        w = dict(charb=0.6, ssim=0.35, edge=0.10, percep=0.0, lpips=0.00, tv=0.01, color=0.00)
        # w = dict(charb=0.5, ssim=0.3, edge=0.15,percep=0.1, lpips=0.05, tv=0.02,color=0.0)
        return CombinedLoss(weights=w, use_mask=False)
    else:
        raise ValueError(f"Unknown preset: {preset}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Restormer import TransformerBlock, Restormer
import math
import pywt

# ====== 基础卷积模块 ======
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ====== 通用 U-Net ======
class UNetResidualPredictor(nn.Module):
    def __init__(self, in_ch=6, base_ch=64, out_ch=3):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(in_ch, base_ch), ConvBlock(base_ch, base_ch))
        self.enc2 = nn.Sequential(ConvBlock(base_ch, base_ch * 2), ConvBlock(base_ch * 2, base_ch * 2))
        self.enc3 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4), ConvBlock(base_ch * 4, base_ch * 4))
        self.bottleneck = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 8), ConvBlock(base_ch * 8, base_ch * 4))
        self.dec3 = nn.Sequential(ConvBlock(base_ch * 8, base_ch * 4), ConvBlock(base_ch * 4, base_ch * 2))
        self.dec2 = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 2), ConvBlock(base_ch * 2, base_ch))
        self.dec1 = nn.Sequential(ConvBlock(base_ch * 2, base_ch), ConvBlock(base_ch, base_ch))
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        return self.out_conv(d1)


# ====== 简化 BaseNet ======
class TinyBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


def extract(a, t, x_shape):
    bs = t.shape[0]
    out = a.gather(-1, t).float().reshape(bs, *((1,) * (len(x_shape) - 1)))
    return out


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size=None, channels=3, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # === 定义噪声调度参数 ===
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    # ============================================================
    # ===============   q_sample: 生成带噪版本   =================
    # ============================================================
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ============================================================
    # ===============   forward: 计算训练损失   ==================
    # ============================================================
    def forward(self, hazy_aug, residual_gt, t):
        """
        用于训练时计算扩散 MSE 损失
        hazy_aug: 条件输入 [B, 67, H, W]
        residual_gt: 真实残差 [B, 3, H, W]
        t: 时间步 [B]
        """
        noise = torch.randn_like(residual_gt)
        x_noisy = self.q_sample(residual_gt, t, noise)
        cond = torch.cat([hazy_aug, x_noisy], dim=1)
        pred_residual = self.model(cond)
        return F.mse_loss(pred_residual, residual_gt)

    # ============================================================
    # ===============   predict: 仅前向预测残差   ================
    # ============================================================
    # @torch.no_grad()
    def predict(self, hazy_aug, t):
        """
        用于训练阶段（RestormerDiffusion.training=True）
        不计算loss，仅预测残差。
        hazy_aug: [B, 67, H, W]
        """
        B, _, H, W = hazy_aug.shape
        device = hazy_aug.device
        z = torch.randn(B, 3, H, W, device=device)
        cond = torch.cat([hazy_aug, z], dim=1)
        residual_pred = self.model(cond)
        return residual_pred

    # ============================================================
    # ===============   sample: 推理采样恢复图像   ===============
    # ============================================================
    @torch.no_grad()
    def sample(self, hazy_aug, steps=25):
        """
        用于推理阶段的采样（简化版本）
        hazy_aug: [B, 67, H, W]
        """
        B, _, H, W = hazy_aug.shape
        device = hazy_aug.device
        z = torch.randn(B, 3, H, W, device=device)

        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            cond = torch.cat([hazy_aug, z], dim=1)
            pred_residual = self.model(cond)
            # 简化为直接更新 z
            z = pred_residual

        return pred_residual


# === Charbonnier loss ===
def charbonnier_loss(x, y, epsilon=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + epsilon ** 2))

# === SSIM loss ===
def ssim_loss(x, y):
    return 1 - ssim_fn(x, y, data_range=1.0, size_average=True)

# === TV loss ===
def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


class RestormerRefineNet(nn.Module):
    def __init__(self, in_ch=6, base_ch=32, num_blocks=1, num_heads=2, expansion_factor=2.66, bias=False, norm_type='BiasFree'):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(
                dim=base_ch,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                bias=bias,
                LayerNorm_type=norm_type
            ) for _ in range(num_blocks)
        ])

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.transformer_blocks(x)
        return self.out_conv(x)



import pywt

# ====== 多尺度小波高频子带提取 ======
def extract_multiscale_wavelet(img, levels=2, wave='haar'):
    B, C, H, W = img.shape
    img_np = img.detach().cpu().numpy()
    wavelet_feats = []

    for i in range(B):
        img_feats = []
        for c in range(C):
            coeffs = pywt.wavedec2(img_np[i, c], wavelet=wave, level=levels)
            for lvl in range(1, levels + 1):
                LH, HL, HH = coeffs[lvl]
                for comp in [LH, HL, HH]:
                    comp_tensor = torch.tensor(comp).unsqueeze(0).unsqueeze(0).float()  # [1,1,h,w]
                    comp_tensor = F.interpolate(comp_tensor, size=(H, W), mode='bilinear', align_corners=False)
                    img_feats.append(comp_tensor)  # 共 3 × levels 个通道
        img_feats_tensor = torch.cat(img_feats, dim=1)  # shape: [1, freq_ch, H, W]
        wavelet_feats.append(img_feats_tensor)

    out = torch.cat(wavelet_feats, dim=0)  # shape: [B, freq_ch, H, W]
    return out.to(img.device)

# ====== 多尺度门控模块：Residual-aware Frequency Gating ======
class ResidualFrequencyGating(nn.Module):
    def __init__(self, img_ch=3, freq_ch=18):  # 注意：3通道图 + 多尺度小波特征
        super().__init__()
        self.res_conv = nn.Conv2d(img_ch, 32, 3, padding=1)
        self.freq_conv = nn.Conv2d(freq_ch, 32, 3, padding=1)
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, freq_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, residual, freq_feat):
        r_feat = self.res_conv(residual)
        f_feat = self.freq_conv(freq_feat)
        alpha = self.gate(r_feat + f_feat)
        return freq_feat * alpha

# ====== Refine模块（Restormer结构） ======
class WRRefineNet(nn.Module):
    def __init__(self, in_ch=3+3+18, base_ch=32, num_blocks=1, num_heads=2, expansion_factor=2.66, bias=False, norm_type='BiasFree'):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(
                dim=base_ch,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                bias=bias,
                LayerNorm_type=norm_type
            ) for _ in range(num_blocks)
        ])
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.transformer_blocks(x)
        return self.out_conv(x)



class ReverseNet(nn.Module):
    """反向路径：base_out → hazy，用于信息保持监督"""
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)  # 输出为 hazy
        )

    def forward(self, x):
        return self.net(x)


from torchvision.models.vgg import vgg16, VGG16_Weights

# ==== 可选：结构感知感知损失 ====
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_vgg = self.vgg(pred)
        target_vgg = self.vgg(target)
        return self.criterion(pred_vgg, target_vgg)

# ==== 结构掩码 ====
lap_kernel = torch.tensor(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]], dtype=torch.float32
).view(1, 1, 3, 3)

def compute_edge_mask(img, threshold=0.05):
    # img: [B, 3, H, W] → 灰度 → 边缘强度
    gray = img.mean(dim=1, keepdim=True)
    lap = torch.abs(F.conv2d(gray, lap_kernel.to(img.device), padding=1))
    mask = (lap > threshold).float()  # [B, 1, H, W]
    return mask


# ==== 反向网络 ====
class ReverseNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)



# ========= 可学起点融合 =========
class StartBlender(nn.Module):
    """
    学习一个逐像素 alpha ∈ [0,1] 来融合 hazy 与 base：
      x0 = alpha * hazy + (1 - alpha) * base
    当 base 质量差时，alpha → 1；当 base 质量好时，alpha → 0。
    """
    def __init__(self, in_ch=6):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(16,  1, 1), nn.Sigmoid()
        )

    def forward(self, hazy, base):
        alpha = self.mask(torch.cat([hazy, base], dim=1))  # [B,1,H,W]
        x0 = alpha * hazy + (1.0 - alpha) * base
        return x0, alpha


# ========= 自适应步长（逐步） =========
class StepScaler(nn.Module):
    """
    根据当前 cond 估计步长 s ∈ [0, max_scale]，误差大时放大步长，误差小则缩小。
    """
    def __init__(self, in_ch, max_scale=2.0):
        super().__init__()
        self.max_scale = max_scale
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, cond):
        s = self.head(cond) * self.max_scale
        return s


# ========= 相对改进损失 =========
def relative_improve_loss(x, base, gt, margin=0.0, p=1):
    """
    要求预测 x 相比 base 更接近 gt。
    使用 Lp 距离（默认 L1），并带 margin（通常设 0 即可）。
    """
    if p == 1:
        d_x    = (x   - gt).abs().mean(dim=(1,2,3))
        d_base = (base - gt).abs().mean(dim=(1,2,3))
    elif p == 2:
        d_x    = ((x   - gt)**2).mean(dim=(1,2,3)).sqrt()
        d_base = ((base - gt)**2).mean(dim=(1,2,3)).sqrt()
    else:
        raise ValueError("p must be 1 or 2")

    rel = torch.clamp(d_x - d_base + margin, min=0.0)
    return rel.mean()



from .Restormer import OverlapPatchEmbed, Downsample, Upsample, LayerNorm, FeedForward, Attention, TransformerBlock
class MiniRestormerRefineNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=32,
                 num_levels=2, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='BiasFree'):
        super().__init__()
        assert num_levels in [1, 2], "Only num_levels=1 or 2 is supported."

        # === 自动设定结构 ===
        if num_levels == 1:
            num_blocks = [1]
            heads = [1]
        else:  # num_levels == 2
            num_blocks = [1, 1]
            heads = [1, 2]

        self.num_levels = num_levels
        self.patch_embed = OverlapPatchEmbed(in_ch, dim)

        # === Encoder ===
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(num_levels):
            level_dim = dim * (2 ** i)
            blocks = [TransformerBlock(level_dim, heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                      for _ in range(num_blocks[i])]
            self.encoders.append(nn.Sequential(*blocks))
            if i < num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        # === Latent ===
        latent_dim = dim * (2 ** (num_levels - 1))
        self.latent = TransformerBlock(latent_dim, heads[-1], ffn_expansion_factor, bias, LayerNorm_type)

        # === Decoder ===
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)
            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, 1, bias=bias))
            blocks = [TransformerBlock(out_dim, heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                      for _ in range(num_blocks[i])]
            self.decoders.append(nn.Sequential(*blocks))

        # === Refinement ===
        self.refinement = nn.Sequential(
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
        )

        self.output = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        inp = x
        feats = []

        x = self.patch_embed(x)
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        x = self.latent(x)

        for i in reversed(range(self.num_levels - 1)):
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)
            x = self.decoders[self.num_levels - 2 - i](x)

        x = self.refinement(x)
        return self.output(x)


# hazy → gt + base作为知识蒸馏      
class KDDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1, lambda_kd=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.is_kd = True  # 作为标志

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 , out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            base_out = self.teacher(hazy)  # 作为 teacher 输出

        x = hazy
        for i in range(self.steps):
            delta = self.refine_steps[i](x)
            x = x + delta

        if training and gt_clear is not None:
            loss_gt = F.mse_loss(x, gt_clear)
            loss_kd = F.mse_loss(x, base_out.detach())  # 与 teacher 对齐
            total_loss = loss_gt + self.lambda_kd * loss_kd
            return total_loss
        else:
            return torch.clamp(x, 0, 1)
        

class KDDiffusion_EM_residual(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1,
                 lambda_kd=0.5, use_edge_mask=False):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False):
        with torch.no_grad():
            teacher_out = self.teacher(hazy)

        x = hazy
        for i in range(self.steps):
            delta = self.refine_steps[i](x)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, teacher_out
        return out

    # def compute_loss(self, hazy, gt_clear, criterion):
    #     """
    #     criterion: CombinedLoss 实例
    #     """
    #     pred, teacher_out = self.forward(hazy, return_teacher=True)

    #     # === 1. 学生 vs GT (主loss: CombinedLoss) ===
    #     total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

    #     # === 2. 学生 vs Teacher (KD loss, edge-aware) ===
    #     if self.use_edge_mask:
    #         edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
    #         kd_loss = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
    #     else:
    #         kd_loss = F.mse_loss(pred, teacher_out.detach())

    #     total_loss = total_loss + self.lambda_kd * kd_loss
    #     parts["kd"] = float(kd_loss.item()) * self.lambda_kd

    #     return total_loss, parts
    def compute_loss(self, hazy, gt_clear, criterion):
        """
        criterion: CombinedLoss 实例
        """
        # 前向：输出 + 教师输出
        pred, teacher_out = self.forward(hazy, return_teacher=True)

        # === 1. 主任务损失：GT对齐 ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. KD：输出层对齐（可加 edge mask）===
        if self.use_edge_mask:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd_out_loss = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
        else:
            kd_out_loss = F.mse_loss(pred, teacher_out.detach())

        # === 3. KD：残差对齐 ===
        teacher_res = teacher_out.detach() - hazy
        student_res = pred - hazy
        kd_res_loss = F.mse_loss(student_res, teacher_res)

        # === 总蒸馏损失（残差 + 输出） ===
        total_kd_loss = self.lambda_kd * (kd_out_loss + kd_res_loss)
        total_loss += total_kd_loss

        # === 记录各项损失 ===
        parts["kd_out"] = float(kd_out_loss.item()) * self.lambda_kd
        parts["kd_res"] = float(kd_res_loss.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())  # 总蒸馏损失

        return total_loss, parts
    
# hazy → gt + base作为知识蒸馏 + step-wise KD
class KDDiffusion_EM_step(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, use_edge_mask=False):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.use_edge_mask = use_edge_mask
        self.is_kd = True  # 标记为蒸馏结构

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False, return_all_steps=False):
        with torch.no_grad():
            teacher_out = self.teacher(hazy)

        x = hazy
        step_outputs = []
        for i in range(self.steps):
            delta = self.refine_steps[i](x)
            x = x + delta
            step_outputs.append(torch.clamp(x, 0, 1))  # 每一步结果都存下来

        out = torch.clamp(x, 0, 1)
        if return_all_steps:
            return out, teacher_out, step_outputs
        elif return_teacher:
            return out, teacher_out
        return out
    # def compute_loss(self, hazy, gt_clear, criterion,kd_criterion=None):
    #     """
    #     criterion: CombinedLoss 实例（用于 GT）
    #     KD 使用 MSE（无需 kd_criterion）
    #     """
    #     pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

    #     # === 1. 主任务损失（对 GT）===
    #     total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

    #     # === 2. Step-wise KD (纯 MSE) ===
    #     kd_total = 0.0
    #     for step_out in step_outputs:
    #         if self.use_edge_mask:
    #             edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
    #             kd_total += F.mse_loss(step_out * edge_mask, teacher_out.detach() * edge_mask)
    #         else:
    #             kd_total += F.mse_loss(step_out, teacher_out.detach())
    #     kd_total /= self.steps

    #     # === 3. 总损失 ===
    #     total_kd_loss = self.lambda_kd * kd_total
    #     total_loss += total_kd_loss

    #     # === 4. 记录各项 ===
    #     parts["kd_stepwise_mse"] = float(kd_total.item()) * self.lambda_kd
    #     parts["kd"] = float(total_kd_loss.item())

    #     return total_loss, parts
# combine loss
    def compute_loss(self, hazy, gt_clear, criterion, kd_criterion=None):
        """
        criterion: CombinedLoss 实例（用于 GT）
        kd_criterion: CombinedLoss 实例（用于 KD）；若为空则 fallback 到 MSE
        """
        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        # === 1. 主任务损失 ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. Step-wise KD (仅输出对齐) ===
        kd_total = 0.0
        kd_parts_accum = {}

        for step_out in step_outputs:
            if kd_criterion is not None:
                step_kd_loss, kd_parts = kd_criterion(step_out, teacher_out, inp_for_mask=hazy)
                kd_total += step_kd_loss
                for k, v in kd_parts.items():
                    kd_parts_accum[k] = kd_parts_accum.get(k, 0.0) + float(v.item())
            else:
                if self.use_edge_mask:
                    edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
                    kd_total += F.mse_loss(step_out * edge_mask, teacher_out.detach() * edge_mask)
                else:
                    kd_total += F.mse_loss(step_out, teacher_out.detach())

        kd_total /= self.steps
        total_kd_loss = self.lambda_kd * kd_total
        total_loss += total_kd_loss

        # === 3. 记录损失 ===
        if kd_criterion is not None:
            for k, v in kd_parts_accum.items():
                parts[f"kd_{k}"] = (v / self.steps) * self.lambda_kd

        parts["kd_stepwise"] = float(kd_total.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())

        return total_loss, parts
    
# hazy → gt + base作为知识蒸馏 + 频域门控
class WaveletKDDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1,
                 lambda_kd=0.5, levels=2):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.is_kd = True  # 蒸馏模型标志

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        # self.refine_steps = nn.ModuleList([
        #     UNetResidualPredictor(in_ch=3 + self.freq_ch) for _ in range(self.steps)
        # ])
        self.refine_steps = nn.ModuleList([
            # WRRefineNet(in_ch=3 + self.freq_ch) for _ in range(self.steps)
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])


    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            base_out = self.teacher(hazy)  # teacher 提供输出
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)  # [B, freq_ch, H, W]

        # === 学生网络 refinement ===
        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        # === Loss ===
        if training and gt_clear is not None:
            loss_gt = F.mse_loss(x, gt_clear)
            loss_kd = F.mse_loss(x, base_out.detach())
            total_loss = loss_gt + self.lambda_kd * loss_kd
            return total_loss
        else:
            return torch.clamp(x, 0, 1)
        
class WaveletKDDiffusion_EM(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False):
        # teacher forward
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)  # [B,freq_ch,H,W]

        # 学生网络 refinement
        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, base_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion):
        """
        criterion: CombinedLoss 实例
        """
        pred, teacher_out = self.forward(hazy, return_teacher=True)

        # === 1. 学生 vs GT (CombinedLoss) ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. KD Loss (edge-aware 可选) ===
        if self.use_edge_mask:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd_loss = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
        else:
            kd_loss = F.mse_loss(pred, teacher_out.detach())

        total_loss = total_loss + self.lambda_kd * kd_loss
        parts["kd"] = float(kd_loss.item()) * self.lambda_kd

        return total_loss, parts

# 残差蒸馏 + 频域门控
class WaveletKDDiffusion_EM_residual(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False):
        # teacher forward
        with torch.no_grad():
            base_out = self.teacher(hazy)
            # 提取多尺度小波
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)  # [B,freq_ch,H,W]

        # 学生网络 refinement
        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, base_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion):
        """
        criterion: CombinedLoss 实例
        """
        pred, teacher_out = self.forward(hazy, return_teacher=True)

        # === 1. 学生 vs GT (CombinedLoss) ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. KD：输出层对齐（可加 edge mask）===
        if self.use_edge_mask:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd_out_loss = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
        else:
            kd_out_loss = F.mse_loss(pred, teacher_out.detach())

        # === 3. KD：残差对齐 ===
        teacher_res = teacher_out.detach() - hazy
        student_res = pred - hazy
        kd_res_loss = F.mse_loss(student_res, teacher_res)

        # === 4. 总蒸馏损失 ===
        total_kd_loss = self.lambda_kd * (kd_out_loss + kd_res_loss)
        total_loss += total_kd_loss

        # === 记录各项损失 ===
        parts["kd_out"] = float(kd_out_loss.item()) * self.lambda_kd
        parts["kd_res"] = float(kd_res_loss.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())  # 总蒸馏损失

        return total_loss, parts
    

# =========================
# 工具函数：生成内窥镜风格的 on-the-fly 烟雾
# =========================
import random
def _perlin_fbm_like(h, w, octaves=3, base_scale=64, device='cuda', dtype=torch.float32):
    """训练友好的 Perlin/FBM 近似：多层随机噪声下采样/上采样叠加"""
    x = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    amp, scale = 1.0, base_scale
    for _ in range(octaves):
        hh = max(1, h // scale)
        ww = max(1, w // scale)
        n = torch.rand(1, 1, hh, ww, device=device, dtype=dtype)
        n = F.interpolate(n, size=(h, w), mode='bilinear', align_corners=False)
        x = x + amp * n
        amp *= 0.5
        scale = max(1, scale // 2)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x  # [1,1,H,W] in [0,1]

def _radial_light(h, w, cx=None, cy=None, r_ratio=0.6, device='cuda', dtype=torch.float32):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    cx = int(w/2) if cx is None else cx
    cy = int(h/2) if cy is None else cy
    rr = ((xx - cx)**2 + (yy - cy)**2).float().sqrt()
    r0 = r_ratio * max(h, w) / 2.0
    L = 1.0 / (1.0 + (rr / (r0 + 1e-6))**2)
    L = (L - L.min()) / (L.max() - L.min() + 1e-6)
    return L[None, None].to(dtype)  # [1,1,H,W]

def _vignette(h, w, power=2.0, device='cuda', dtype=torch.float32):
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing='ij')
    r = torch.sqrt(xx**2 + yy**2).clamp(0,1)
    V = (1 - r)**power
    return V[None, None].to(dtype)  # [1,1,H,W]

def _edge_suppression(img):
    """img: [B,3,H,W], [0,1] -> 返回 [B,1,H,W] 用于保护边缘不被完全烟雾化"""
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=img.device, dtype=img.dtype).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)
    g = 0
    for c in range(img.shape[1]):
        ch = img[:, c:c+1]
        gx = F.conv2d(ch, sobel_x, padding=1)
        gy = F.conv2d(ch, sobel_y, padding=1)
        g += (gx**2 + gy**2).sqrt()
    g = g / img.shape[1]
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return g  # [B,1,H,W]

import torch
import torch.nn.functional as F
import random
import math
from torch import Tensor

@torch.no_grad()
def add_smoke_endoscopic_onfly(J: Tensor, *,
                               epoch: int = None,
                               start_epoch: int = 0,
                               use_dynamic_alpha: bool = False,
                               fixed_alpha: float = 1.0,
                               return_alpha: bool = False,
                               beta=(0.8, 1.6),
                               wp=0.8, wl=0.4, wv=0.2, we=0.2,
                               a0=(0.90, 0.88, 0.85),
                               alpha0=(0.85, 0.98),
                               gamma=(0.10, 0.30),
                               glow_prob=0.5,
                               device=None,
                               dtype=torch.float32):
    """
    内窥镜友好的 on-the-fly 加雾（支持延迟启用 + 动态调节 alpha_aug）

    参数：
        J: 清晰图 [B,3,H,W]，取值范围 [0,1]
        epoch: 当前训练轮次（控制 OTF 启动与否）
        start_epoch: 启用 OTF 的起始轮次
        use_dynamic_alpha: 是否动态调节 alpha_aug（余弦调度）
        fixed_alpha: 非动态时使用的固定 alpha 值
        return_alpha: 是否额外返回 alpha_aug（用于动态 loss 调整）
    """

    # === 安全处理：epoch 为 None 时赋默认值 ===
    if epoch is None:
        epoch = start_epoch

    # === 未到启用时机，直接返回原图 ===
    if epoch < start_epoch:
        if return_alpha:
            return J.clone(), {"t": None}, 0.0
        return J.clone(), {"t": None}

    if device is None:
        device = J.device
    J = J.to(device=device, dtype=dtype)
    B, C, H, W = J.shape

    # === 动态调节 alpha_aug ===
    if use_dynamic_alpha:
        max_epoch = 120
        progress = min(float(epoch), float(max_epoch))
        alpha_aug = fixed_alpha * (1 - math.cos(math.pi * progress / max_epoch)) / 2
    else:
        alpha_aug = fixed_alpha

    # === 烟雾结构成分生成 ===
    P = _perlin_fbm_like(H, W, octaves=random.choice([2,3,4]),
                         base_scale=random.choice([32,48,64]),
                         device=device, dtype=dtype)
    L = _radial_light(H, W, r_ratio=random.choice([0.5,0.6,0.7]),
                      device=device, dtype=dtype)
    V = _vignette(H, W, power=random.choice([1.5,2.0,2.5]),
                  device=device, dtype=dtype)
    E = _edge_suppression(J)

    D = wp * P + wl * L + wv * V - we * E
    D = D.clamp(0, 1)

    beta_val = torch.empty(B,1,1,1, device=device, dtype=dtype).uniform_(*beta)
    t = torch.exp(-beta_val * D)

    alpha0_val = torch.empty(B,1,1,1, device=device, dtype=dtype).uniform_(*alpha0)
    gamma_val  = torch.empty(B,1,1,1, device=device, dtype=dtype).uniform_(*gamma)
    a0_vec = torch.tensor(a0, device=device, dtype=dtype).view(1,3,1,1)
    A = alpha0_val * a0_vec * (1 + gamma_val * L)
    A = A.clamp(0, 1)

    I = J * t + A * (1 - t)

    # === 可选辉光模拟 ===
    if random.random() < glow_prob:
        tau   = random.uniform(0.7, 0.85)
        kappa = random.uniform(0.05, 0.12)
        bright = (J - tau).clamp(min=0.0)
        sigma_map = 1 + 5 * (1 - t)
        G = 0
        for _ in range(2):
            blur = F.avg_pool2d(bright, kernel_size=5, stride=1, padding=2)
            G = G + blur * sigma_map
        I = (I + kappa * G).clamp(0, 1)

    if return_alpha:
        return I, {"t": t, "D": D, "L": L, "P": P, "A": A}, alpha_aug
    return I, {"t": t, "D": D, "L": L, "P": P, "A": A}

# =========================
# WaveletKDDiffusion_EM 的 OTF 版本
# =========================

# class WaveletKDDiffusion_EM_otf(nn.Module):
#     def __init__(self, base_ckpt_path=None, base_net=None,
#                  steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
#                  # --- OTF 参数 ---
#                  use_onfly=True, p_aug=0.5, alpha_aug=0.5,
#                  otf_beta=(0.8, 1.6),
#                  otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
#                  otf_a0=(0.90, 0.88, 0.85),
#                  otf_alpha0=(0.85, 0.98),
#                  otf_gamma=(0.10, 0.30),
#                  otf_glow_prob=0.5):
#         super().__init__()
#         assert 1 <= steps <= 10
#         self.steps = steps
#         self.lambda_kd = lambda_kd
#         self.levels = levels
#         self.is_kd = True
#         self.use_edge_mask = use_edge_mask

#         # --- OTF 控制 ---
#         self.use_onfly = use_onfly
#         self.p_aug = p_aug
#         self.alpha_aug = alpha_aug
#         self._otf_cfg = dict(
#             beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
#             a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma,
#             glow_prob=otf_glow_prob
#         )

#         # === 教师网络 ===
#         self.teacher = base_net if base_net is not None else TinyBaseNet()
#         if base_ckpt_path is not None:
#             ckpt = torch.load(base_ckpt_path, map_location="cpu")
#             state_dict = ckpt.get("state_dict", ckpt)
#             new_state_dict = {
#                 k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
#                 for k, v in state_dict.items()
#             }
#             self.teacher.load_state_dict(new_state_dict, strict=False)
#         for p in self.teacher.parameters():
#             p.requires_grad = False

#         # === 小波频域 gating 模块 ===
#         self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
#         self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

#         # === 学生网络 ===
#         self.refine_steps = nn.ModuleList([
#             MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
#             for _ in range(self.steps)
#         ])

#     def forward(self, hazy, return_teacher=False):
#         # === 教师分支（无梯度） ===
#         with torch.no_grad():
#             base_out = self.teacher(hazy)
#             freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
#             freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
#             freq_residual = freq_hazy - freq_base
#             residual = hazy - base_out
#             gated_freq = self.freq_gating(residual, freq_residual)  # [B,freq_ch,H,W]

#         # === 学生 refine ===
#         x = hazy
#         for i in range(self.steps):
#             cond = torch.cat([x, gated_freq], dim=1)
#             delta = self.refine_steps[i](cond)
#             x = x + delta

#         out = torch.clamp(x, 0, 1)
#         if return_teacher:
#             return out, base_out
#         return out

#     def _kd_term(self, pred, teacher_out, gt_clear=None):
#         """可选 edge-aware KD"""
#         if self.use_edge_mask and gt_clear is not None:
#             edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
#             kd = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
#         else:
#             kd = F.mse_loss(pred, teacher_out.detach())
#         return kd

#     def compute_loss(self, hazy, gt_clear, criterion):
#         """
#         criterion: CombinedLoss 实例
#         主分支：真实 hazy → pred，对齐 gt_clear + KD
#         增强分支：随机对 gt_clear on-the-fly 加雾 → pred_aug，对齐同一个 gt_clear + KD
#         """
#         # --- 主分支 ---
#         pred, teacher_out = self.forward(hazy, return_teacher=True)
#         total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

#         kd_loss = self._kd_term(pred, teacher_out, gt_clear=gt_clear)
#         total_loss = total_loss + self.lambda_kd * kd_loss
#         parts["kd"] = float(kd_loss.item()) * self.lambda_kd

#         # --- 增强分支（随机触发） ---
#         if self.use_onfly and random.random() < self.p_aug:
#             with torch.no_grad():
#                 hazy_aug, _meta = add_smoke_endoscopic_onfly(
#                     gt_clear,
#                     device=gt_clear.device,
#                     dtype=gt_clear.dtype,
#                     **self._otf_cfg
#                 )
#             pred_aug, teacher_aug = self.forward(hazy_aug, return_teacher=True)

#             loss_aug, _ = criterion(pred_aug, gt_clear, inp_for_mask=hazy_aug)
#             kd_aug = self._kd_term(pred_aug, teacher_aug, gt_clear=gt_clear)

#             aug_term = loss_aug + self.lambda_kd * kd_aug
#             total_loss = total_loss + self.alpha_aug * aug_term

#             parts["otf_loss"] = float(loss_aug.item()) * self.alpha_aug
#             parts["otf_kd"]   = float(kd_aug.item())   * self.alpha_aug

#         return total_loss, parts




# 频域门控 + step-wise KD
class WaveletKDDiffusion_EM_step(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False, return_all_steps=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        step_outputs = []
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta
            step_outputs.append(x)  # 每一步的输出

        out = torch.clamp(x, 0, 1)
        if return_all_steps:
            return out, base_out, step_outputs
        elif return_teacher:
            return out, base_out
        return out

    # def compute_loss(self, hazy, gt_clear, criterion,kd_criterion=None):
    #     """
    #     criterion: CombinedLoss 实例
    #     """
    #     pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

    #     # === 1. GT loss ===
    #     total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

    #     # === 2. Step-wise KD ===
    #     kd_step_loss = 0
    #     for step_out in step_outputs:
    #         if self.use_edge_mask:
    #             edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
    #             kd_step_loss += F.mse_loss(step_out * edge_mask, teacher_out.detach() * edge_mask)
    #         else:
    #             kd_step_loss += F.mse_loss(step_out, teacher_out.detach())
    #     kd_step_loss /= self.steps  # 求平均

    #     # === 3. 总损失 ===
    #     total_kd_loss = self.lambda_kd * kd_step_loss
    #     total_loss += total_kd_loss

    #     parts["kd_stepwise"] = float(kd_step_loss.item()) * self.lambda_kd
    #     parts["kd"] = float(total_kd_loss.item())

    #     return total_loss, parts
    # Combinedloss 
    def compute_loss(self, hazy, gt_clear, criterion, kd_criterion=None):
        """
        criterion     : CombinedLoss 实例（对 GT）
        kd_criterion  : CombinedLoss 实例（对 teacher 蒸馏）
        """
        assert kd_criterion is not None, "Please provide kd_criterion as a CombinedLoss instance."

        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        # === 1. GT loss ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. Step-wise KD (感知增强) ===
        kd_total = 0
        kd_parts_accumulate = {}  # 用于每项 loss 求平均

        for step_out in step_outputs:
            step_loss, step_parts = kd_criterion(step_out, teacher_out, inp_for_mask=hazy)
            kd_total += step_loss

            # 汇总每一项 loss 的值
            for k, v in step_parts.items():
                kd_parts_accumulate[k] = kd_parts_accumulate.get(k, 0.0) + float(v.item())

        kd_total = kd_total / self.steps
        total_loss += self.lambda_kd * kd_total

        # === 3. 记录 KD 每项 ===
        for k, v_sum in kd_parts_accumulate.items():
            parts[f"kd_{k}"] = (v_sum / self.steps) * self.lambda_kd
        parts["kd_stepwise"] = float(kd_total.item()) * self.lambda_kd
        parts["kd"] = float((self.lambda_kd * kd_total).item())

        return total_loss, parts
# === 是否启用 OTF 的判断函数 ===
def should_enable_otf(epoch, otf_start_epoch=10):
    return epoch >= otf_start_epoch


class WaveletKDDiffusion_EM_otf_noKD(nn.Module):
    """
    无蒸馏（non-KD）对照结构：
      - 保留小波频域 gating + 多步 refine + OTF smoke augmentation
      - 移除 teacher 网络和蒸馏项
      - 用于验证蒸馏的性能提升效果
    """
    def __init__(self,
                 steps=1, levels=2,
                 # --- OTF 参数 ---
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 otf_beta=(0.8, 1.6),
                 otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85),
                 otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30),
                 otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.levels = levels
        self.is_kd = False  # 明确标注：无蒸馏结构

        # --- OTF 控制 ---
        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.use_dynamic_alpha = use_dynamic_alpha
        self._otf_cfg = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma,
            glow_prob=otf_glow_prob
        )

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络（保留 refine 逻辑） ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy):
        # === 无教师分支 ===
        freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
        # 无 base_out，因此 freq_residual=0，直接送 gating
        zero_like = torch.zeros_like(freq_hazy)
        gated_freq = self.freq_gating(hazy, zero_like)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        return out

    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        """
        criterion: CombinedLoss 实例
        主分支：真实 hazy → pred，对齐 gt_clear
        增强分支：随机对 gt_clear on-the-fly 加雾 → pred_aug，对齐同一个 gt_clear
        """
        # --- 主分支 ---
        pred = self.forward(hazy)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # --- OTF 增强分支 ---
        if self.use_onfly and (epoch is not None and should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_aug, _meta = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self._otf_cfg
                )
            pred_aug = self.forward(hazy_aug)
            loss_aug, _ = criterion(pred_aug, gt_clear, inp_for_mask=hazy_aug)
            total_loss = total_loss + self.alpha_aug * loss_aug
            parts["otf_loss"] = float(loss_aug.item()) * self.alpha_aug

        return total_loss, parts



# ============ WaveletKDDiffusion_EM_otf (MSE KD版) ============ #
class WaveletKDDiffusion_EM_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
                 # --- OTF 参数 ---
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 otf_beta=(0.8, 1.6),
                 otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85),
                 otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30),
                 otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        # --- OTF 控制 ---
        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.use_dynamic_alpha = use_dynamic_alpha
        self._otf_cfg = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma,
            glow_prob=otf_glow_prob
        )

        # === 教师网络（冻结） ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    # -------------------------
    def forward(self, hazy, return_teacher=False):
        """前向传播"""
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, base_out
        return out

    # -------------------------
    def _kd_term(self, pred, teacher_out, gt_clear=None):
        """使用 MSE 蒸馏损失"""
        with torch.no_grad():
            t_out = teacher_out.detach()

        if self.use_edge_mask and gt_clear is not None:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd = F.mse_loss(pred * edge_mask, t_out * edge_mask)
        else:
            kd = F.mse_loss(pred, t_out)
        return kd

    # -------------------------
    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        """
        criterion: CombinedLoss 实例
        主分支：真实 hazy → pred，对齐 gt_clear + KD
        增强分支：随机对 gt_clear on-the-fly 加雾 → pred_aug，对齐同一个 gt_clear + KD
        """
        # --- 主分支 ---
        pred, teacher_out = self.forward(hazy, return_teacher=True)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        kd_loss = self._kd_term(pred, teacher_out, gt_clear=gt_clear)
        total_loss = total_loss + self.lambda_kd * kd_loss
        parts["kd"] = float(kd_loss.item()) * self.lambda_kd

        # --- OTF 增强分支 ---
        if self.use_onfly and (epoch is not None and should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_aug, _meta = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self._otf_cfg
                )

            pred_aug, teacher_aug = self.forward(hazy_aug, return_teacher=True)
            loss_aug, _ = criterion(pred_aug, gt_clear, inp_for_mask=hazy_aug)
            kd_aug = self._kd_term(pred_aug, teacher_aug, gt_clear=gt_clear)

            aug_term = loss_aug + self.lambda_kd * kd_aug
            total_loss = total_loss + self.alpha_aug * aug_term

            parts["otf_loss"] = float(loss_aug.item()) * self.alpha_aug
            parts["otf_kd"]   = float(kd_aug.item())   * self.alpha_aug

        return total_loss, parts



class KDDiffusion_EM_step_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, use_edge_mask=False,
                 # ==== OTF 控制 ====
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 # ==== OTF 参数 ====
                 otf_beta=(0.8, 1.6),
                 otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85),
                 otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30),
                 otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.use_dynamic_alpha = use_dynamic_alpha
        self.use_edge_mask = use_edge_mask
        self.is_kd = True

        # ==== OTF 参数 ====
        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        # === 教师网络 ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False, return_all_steps=False):
        with torch.no_grad():
            teacher_out = self.teacher(hazy)

        x = hazy
        step_outputs = []
        for i in range(self.steps):
            delta = self.refine_steps[i](x)
            x = x + delta
            step_outputs.append(torch.clamp(x, 0, 1))
        out = torch.clamp(x, 0, 1)

        if return_all_steps:
            return out, teacher_out, step_outputs
        elif return_teacher:
            return out, teacher_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion, kd_criterion=None, epoch=None):
        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        # === 主任务 ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === Step-wise 蒸馏 ===
        kd_total = 0.0
        kd_parts_accum = {}
        kd_loss_fn = kd_criterion if kd_criterion is not None else criterion

        for step_out in step_outputs:
            kd_loss, kd_parts = kd_loss_fn(step_out, teacher_out.detach(), inp_for_mask=hazy)
            kd_total += kd_loss
            for k, v in kd_parts.items():
                kd_parts_accum[k] = kd_parts_accum.get(k, 0.0) + float(v.item())

        kd_total /= self.steps
        total_kd_loss = self.lambda_kd * kd_total
        total_loss += total_kd_loss

        for k, v in kd_parts_accum.items():
            parts[f"kd_{k}"] = (v / self.steps) * self.lambda_kd
        parts["kd_stepwise"] = float(kd_total.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())

        # === OTF 动态增强（受 epoch 控制） ===
        if self.use_onfly and (epoch is not None and should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self.otf_kwargs
                )
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts


    

class WaveletKDDiffusion_EM_step_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5,use_dynamic_alpha=False,
                 # ==== OTF 参数 ====
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.use_dynamic_alpha = use_dynamic_alpha
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False, return_all_steps=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        step_outputs = []
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta
            step_outputs.append(x)

        out = torch.clamp(x, 0, 1)
        if return_all_steps:
            return out, base_out, step_outputs
        elif return_teacher:
            return out, base_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion, kd_criterion=None, epoch=None):
        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        kd_total = 0.0
        kd_parts_accum = {}
        kd_loss_fn = kd_criterion if kd_criterion is not None else criterion

        for step_out in step_outputs:
            kd_loss, kd_parts = kd_loss_fn(step_out, teacher_out.detach(), inp_for_mask=hazy)
            kd_total += kd_loss
            for k, v in kd_parts.items():
                kd_parts_accum[k] = kd_parts_accum.get(k, 0.0) + float(v.item())

        kd_total /= self.steps
        total_kd_loss = self.lambda_kd * kd_total
        total_loss += total_kd_loss

        for k, v in kd_parts_accum.items():
            parts[f"kd_{k}"] = (v / self.steps) * self.lambda_kd
        parts["kd_stepwise"] = float(kd_total.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())

        if self.use_onfly and (epoch is not None and should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self.otf_kwargs
                )
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts





# 对比损失函数：NT-Xent Loss

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

class KDDiffusion_EM_contrast_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_contrast=0.5,
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5,use_dynamic_alpha=False,
                 # ==== OTF 参数 ====
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_contrast = lambda_contrast
        self.use_onfly = use_onfly
        self.use_dynamic_alpha = use_dynamic_alpha
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.is_kd = True

        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, hazy, return_teacher=False, return_feature=False):
        with torch.no_grad():
            teacher_out = self.teacher(hazy)

        x = hazy
        for i in range(self.steps):
            delta = self.refine_steps[i](x)
            x = x + delta
        out = torch.clamp(x, 0, 1)

        if return_feature:
            feat_s = self.proj_head(self.feature_extractor(out))
            with torch.no_grad():
                feat_t = self.proj_head(self.feature_extractor(teacher_out))
            return out, teacher_out, feat_s, feat_t

        if return_teacher:
            return out, teacher_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        pred, teacher_out, feat_s, feat_t = self.forward(hazy, return_feature=True)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # 对比损失
        contrast_loss = nt_xent_loss(feat_s, feat_t)
        total_loss += self.lambda_contrast * contrast_loss
        parts["contrast"] = float(contrast_loss.item()) * self.lambda_contrast

        # On-the-fly 烟雾增强分支
        if self.use_onfly and (epoch is None or should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(gt_clear, device=gt_clear.device, use_dynamic_alpha=self.use_dynamic_alpha,dtype=gt_clear.dtype, **self.otf_kwargs)
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts



class WaveletKDDiffusion_EM_contrast_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_contrast=0.5, levels=2,
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5,use_dynamic_alpha=False,
                 # ==== OTF 参数 ====
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_contrast = lambda_contrast
        self.levels = levels
        self.use_onfly = use_onfly
        self.use_dynamic_alpha = use_dynamic_alpha
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.is_kd = True

        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, hazy, return_teacher=False, return_feature=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta
        out = torch.clamp(x, 0, 1)

        if return_feature:
            feat_s = self.proj_head(self.feature_extractor(out))
            with torch.no_grad():
                feat_t = self.proj_head(self.feature_extractor(base_out))
            return out, base_out, feat_s, feat_t

        if return_teacher:
            return out, base_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        pred, teacher_out, feat_s, feat_t = self.forward(hazy, return_feature=True)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        contrast_loss = nt_xent_loss(feat_s, feat_t)
        total_loss += self.lambda_contrast * contrast_loss
        parts["contrast"] = float(contrast_loss.item()) * self.lambda_contrast

        if self.use_onfly and (epoch is None or should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(gt_clear, device=gt_clear.device, use_dynamic_alpha=self.use_dynamic_alpha,dtype=gt_clear.dtype, **self.otf_kwargs)
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts



class WaveletKDDiffusion_EM_dynamic_contrast_otf(nn.Module):
    """
    动态对比蒸馏：
      - 前期：对比损失为主（lambda_contrast 高），KD Charb 低/为0
      - 中后期：逐步降低对比损失，升高KD Charb（朝向教师输出但更贴合GT方向的稳定蒸馏）
      - 全程：主监督由外部 `criterion`(CombinedLoss) 对 pred vs GT 负责；OTF 按 epoch 开启
    """
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, levels=2,
                 # ===== 动态权重调度参数 =====
                 lambda_contrast_max=0.6,      # 早期对比学习的最大权重
                 lambda_kd_max=0.6,            # 后期KD(Charb)的最大权重
                 contrast_full_epochs=60,      # 前 contrast_full_epochs 维持最大对比权重
                 contrast_fade_epochs=20,      # 随后 contrast_fade_epochs 线性衰减到 0
                 kd_warmup_epochs=20,          # 在 contrast_full_epochs 之后，KD 从0线性升至 lambda_kd_max
                 use_cosine_schedule=False,    # True 则用余弦而非线性
                 # ===== OTF 参数 =====
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.levels = levels

        # ===== 动态权重调度配置 =====
        self.lambda_contrast_max = float(lambda_contrast_max)
        self.lambda_kd_max = float(lambda_kd_max)
        self.contrast_full_epochs = int(contrast_full_epochs)
        self.contrast_fade_epochs = int(contrast_fade_epochs)
        self.kd_warmup_epochs = int(kd_warmup_epochs)
        self.use_cosine_schedule = bool(use_cosine_schedule)

        # ===== OTF 配置 =====
        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.use_dynamic_alpha = use_dynamic_alpha
        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        # ===== 教师网络（冻结）=====
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ===== 频域 gating =====
        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # ===== 学生多步 refine（非 stepwise KD，只是多次细化）=====
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        # ===== 对比学习特征头（学生与教师共享提取器+投影头参数，不共享梯度）=====
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    # -------- forward --------
    def forward(self, hazy, return_teacher=False, return_feature=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta
        out = torch.clamp(x, 0, 1)

        if return_feature:
            feat_s = self._project_feature(out)
            with torch.no_grad():
                feat_t = self._project_feature(base_out)
            return out, base_out, feat_s, feat_t

        if return_teacher:
            return out, base_out
        return out

    def _project_feature(self, img_3ch):
        feat = self.feature_extractor(img_3ch)
        z = self.proj_head(feat)
        return z

    # -------- 损失函数组件 --------
    @staticmethod
    def _charbonnier_loss(x, y, eps=1e-6):
        return torch.mean(torch.sqrt((x - y) ** 2 + eps))

    # -------- 动态权重调度（线性或余弦）--------
    def _schedule_weights(self, epoch: int):
        """
        返回 (lambda_contrast, lambda_kd) 随 epoch 的动态权重
        时间线：
          [0, contrast_full_epochs): contrast = max, kd = 0
          [contrast_full_epochs, contrast_full_epochs + contrast_fade_epochs):
                contrast 线性(或余弦)衰减到 0
                kd       同步线性(或余弦)上升到 lambda_kd_max（历时 kd_warmup_epochs）
          之后：contrast=0, kd=lambda_kd_max
        """
        e = max(0, int(epoch))

        # 对比项
        if e < self.contrast_full_epochs:
            l_con = self.lambda_contrast_max
            k = 0.0
        elif e < self.contrast_full_epochs + self.contrast_fade_epochs:
            t = (e - self.contrast_full_epochs) / max(1, self.contrast_fade_epochs)
            if self.use_cosine_schedule:
                # 余弦从 1 -> 0
                decay = 0.5 * (1 + torch.cos(torch.tensor(t * 3.1415926535))).item()
            else:
                decay = 1.0 - t
            l_con = max(0.0, self.lambda_contrast_max * decay)
            # KD 同步启动/升温
            if self.kd_warmup_epochs > 0:
                tw = min(1.0, (e - self.contrast_full_epochs) / self.kd_warmup_epochs)
            else:
                tw = 1.0
            if self.use_cosine_schedule:
                grow = 1.0 - 0.5 * (1 + torch.cos(torch.tensor(tw * 3.1415926535))).item()  # 0->1
            else:
                grow = tw
            k = self.lambda_kd_max * grow
        else:
            l_con = 0.0
            k = self.lambda_kd_max

        return float(l_con), float(k)

    # -------- compute_loss --------
    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        """
        criterion: CombinedLoss 实例（负责 supervised pred vs GT）
        动态项：
          - 对比损失：nt_xent_loss(feat_s, feat_t)，权重 lambda_contrast(epoch)
          - KD(Charb)：Charbonnier(pred, teacher_out)，权重 lambda_kd(epoch)
        OTF：按 should_enable_otf(epoch) 开启；仅加 supervised 分支（必要时也可加 KD/对比，见注释）
        """
        pred, teacher_out, feat_s, feat_t = self.forward(hazy, return_feature=True)

        # 1) 主监督：pred vs GT（你现有的 CombinedLoss 里已包含 Charb/SSIM/Edge 等）
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # 2) 动态权重
        l_contrast, l_kd = self._schedule_weights(epoch if epoch is not None else 0)
        parts["λ_contrast"] = float(l_contrast)
        parts["λ_kd_charb"] = float(l_kd)

        # 3) 对比损失（学生 vs 教师）——早期权重大
        if l_contrast > 0:
            contrast_loss = nt_xent_loss(feat_s, feat_t)
            total_loss = total_loss + l_contrast * contrast_loss
            parts["contrast"] = float(contrast_loss.item()) * l_contrast

        # 4) KD(Charbonnier) —— 后期权重大，强调对齐教师输出的稳定监督（与GT主监督协同）
        if l_kd > 0:
            kd_charb = self._charbonnier_loss(pred, teacher_out.detach())
            total_loss = total_loss + l_kd * kd_charb
            parts["kd_charb"] = float(kd_charb.item()) * l_kd

        # 5) OTF 增强（与动态权重兼容）
        if self.use_onfly and (epoch is None or should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self.otf_kwargs
                )
            # 仅进行 supervised（保持简洁稳定）；如需，可在此处同样加入对比+KD，权重复用 l_contrast/l_kd
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss = total_loss + self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts




# =========================
# 判别器：小波结构级（多尺度高频）
# =========================
class WaveletDiscriminator(nn.Module):
    def __init__(self, in_ch, base_ch=48, n_blocks=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for _ in range(n_blocks - 1):
            layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, inplace=True)]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# 主结构：WaveletKD + MSE蒸馏 + 结构级对抗 + OTF
# =========================
class WaveletKDDiffusion_EM_otf_WaveGAN_mse(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
                 # --- Wavelet-GAN 参数 ---
                 lambda_adv=0.03, adv_on_synth_only=False,
                 use_ms_high=True, use_top_level_only=False,
                 # --- OTF 参数 ---
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.is_kd = True
        self.use_edge_mask = use_edge_mask

        # --- Wavelet-GAN 控制 ---
        self.lambda_adv = lambda_adv
        self.adv_on_synth_only = adv_on_synth_only
        self.use_ms_high = use_ms_high
        self.use_top_level_only = use_top_level_only

        # --- OTF 控制 ---
        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.use_dynamic_alpha = use_dynamic_alpha
        self._otf_cfg = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma,
            glow_prob=otf_glow_prob
        )

        # === 教师网络（冻结） ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生 refine steps ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        # === Wavelet 结构级判别器 ===
        disc_in_ch = 9 * (levels if self.use_ms_high else 1)
        self.discriminator = WaveletDiscriminator(in_ch=disc_in_ch, base_ch=48, n_blocks=3)

    # -------------------------
    # 核心前向
    # -------------------------
    def forward(self, hazy, return_teacher=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, base_out
        return out

    # -------------------------
    # 损失部件
    # -------------------------
    def _kd_term(self, pred, teacher_out, gt_clear=None):
        """用 MSE 替代 Charbonnier"""
        with torch.no_grad():
            t_out = teacher_out.detach()
        if self.use_edge_mask and gt_clear is not None:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd = F.mse_loss(pred * edge_mask, t_out * edge_mask)
        else:
            kd = F.mse_loss(pred, t_out)
        return kd

    def _collect_highfreq(self, freq_tensor):
        B, C, H, W = freq_tensor.shape
        per_level = 9
        if self.use_ms_high:
            if self.use_top_level_only:
                return freq_tensor[:, -per_level:, :, :]
            else:
                return freq_tensor
        else:
            return freq_tensor[:, -per_level:, :, :]

    def _d_loss(self, d_out_fake, d_out_real):
        return F.softplus(d_out_fake).mean() + F.softplus(-d_out_real).mean()

    def _g_loss(self, d_out_fake):
        return F.softplus(-d_out_fake).mean()

    # -------------------------
    # 训练接口：返回 G、D、parts
    # -------------------------
    def compute_loss(self, hazy, gt_clear, criterion, epoch=None, is_synth=False):
        parts = {}
        D_total = torch.tensor(0.0, device=hazy.device)

        # --- 主分支 ---
        pred, teacher_out = self.forward(hazy, return_teacher=True)
        rec_loss, parts_rec = criterion(pred, gt_clear, inp_for_mask=hazy)
        parts.update({f"rec_{k}": float(v) for k, v in parts_rec.items()})

        G_total = rec_loss
        kd = self._kd_term(pred, teacher_out, gt_clear=gt_clear)
        G_total = G_total + self.lambda_kd * kd
        parts["kd"] = float(kd.item()) * self.lambda_kd

        # 结构级对抗
        enable_adv = (self.lambda_adv > 0.0) and (not self.adv_on_synth_only)
        if enable_adv:
            with torch.no_grad():
                freq_gt = extract_multiscale_wavelet(gt_clear, levels=self.levels)
            freq_pred = extract_multiscale_wavelet(pred, levels=self.levels)

            feat_gt = self._collect_highfreq(freq_gt)
            feat_pred = self._collect_highfreq(freq_pred)

            d_fake = self.discriminator(feat_pred.detach())
            d_real = self.discriminator(feat_gt)
            d_loss_main = self._d_loss(d_fake, d_real)

            D_total = D_total + d_loss_main
            parts["d_main"] = float(d_loss_main.item())

            g_adv = self._g_loss(self.discriminator(feat_pred))
            G_total = G_total + self.lambda_adv * g_adv
            parts["g_adv_main"] = float(g_adv.item()) * self.lambda_adv

        # --- OTF 增强分支 ---
        if self.use_onfly and (epoch is not None and should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_aug, _meta = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self._otf_cfg
                )
            pred_aug, teacher_aug = self.forward(hazy_aug, return_teacher=True)
            rec_aug, _ = criterion(pred_aug, gt_clear, inp_for_mask=hazy_aug)
            kd_aug = self._kd_term(pred_aug, teacher_aug, gt_clear=gt_clear)

            aug_gen_term = rec_aug + self.lambda_kd * kd_aug

            enable_adv_aug = (self.lambda_adv > 0.0) and (self.adv_on_synth_only or True)
            if enable_adv_aug:
                with torch.no_grad():
                    freq_gt_aug = extract_multiscale_wavelet(gt_clear, levels=self.levels)
                freq_pred_aug = extract_multiscale_wavelet(pred_aug, levels=self.levels)

                feat_gt_aug = self._collect_highfreq(freq_gt_aug)
                feat_pred_aug = self._collect_highfreq(freq_pred_aug)

                d_fake_aug = self.discriminator(feat_pred_aug.detach())
                d_real_aug = self.discriminator(feat_gt_aug)
                d_loss_aug = self._d_loss(d_fake_aug, d_real_aug)

                D_total = D_total + d_loss_aug
                parts["d_otf"] = float(d_loss_aug.item())

                g_adv_aug = self._g_loss(self.discriminator(feat_pred_aug))
                aug_gen_term = aug_gen_term + self.lambda_adv * g_adv_aug
                parts["g_adv_otf"] = float(g_adv_aug.item()) * self.lambda_adv

            G_total = G_total + self.alpha_aug * aug_gen_term
            parts["otf_rec"] = float(rec_aug.item()) * self.alpha_aug
            parts["otf_kd"]  = float(kd_aug.item()) * self.alpha_aug

        # === 最终总损失（方案 A：合并返回） ===
        total = G_total + D_total

        # 记录方便日志
        parts["total_G"] = float(G_total.item())
        parts["total_D"] = float(D_total.item())
        parts["total"]   = float(total.item())

        return total, parts



# ============================================
# 特征级判别器 (Feature-Level Discriminator)
# ============================================
class FeatureDiscriminator(nn.Module):
    """
    输入: 特征张量 [B, C, H, W]
    用于判断特征是否来自 GT (real) 或 Student (fake)
    """
    def __init__(self, in_ch, base_ch=64, n_blocks=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for _ in range(n_blocks - 1):
            layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, inplace=True)]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ============================================
# 主结构: WaveletKD + MSE蒸馏 + 特征级对抗 + OTF
# ============================================
class WaveletKDDiffusion_EM_otf_FeatureGAN_mse(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
                 # --- Feature-GAN 参数 ---
                 lambda_adv=0.03, adv_on_synth_only=False,
                 feat_extractor=None, feat_dim=64,
                 # --- OTF 参数 ---
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = lambda_kd
        self.levels = levels
        self.use_edge_mask = use_edge_mask

        # --- 对抗控制 ---
        self.lambda_adv = lambda_adv
        self.adv_on_synth_only = adv_on_synth_only
        self.feat_dim = feat_dim

        # --- OTF 控制 ---
        self.use_onfly = use_onfly
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.use_dynamic_alpha = use_dynamic_alpha
        self._otf_cfg = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma,
            glow_prob=otf_glow_prob
        )

        # === 教师网络（冻结） ===
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # === 小波 gating 模块 ===
        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        # === 特征提取器（兼容 TinyBaseNet / Restormer / 其它）===
        if feat_extractor is not None:
            self.feat_extractor = feat_extractor
        elif hasattr(self.teacher, "encoder"):
            self.feat_extractor = self.teacher.encoder
        elif hasattr(self.teacher, "forward_features"):
            self.feat_extractor = self.teacher
        else:
            self.feat_extractor = self.teacher

        for p in self.feat_extractor.parameters():
            p.requires_grad = False

        # === 特征映射层（用于 TinyBaseNet 输出3通道时投影至 feat_dim）===
        self.feat_proj = nn.Conv2d(3, feat_dim, kernel_size=1, bias=False)

        # === 特征判别器 ===
        self.discriminator = FeatureDiscriminator(in_ch=feat_dim, base_ch=64, n_blocks=3)

    # -------------------------
    def forward(self, hazy, return_teacher=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, base_out
        return out

    # -------------------------
    def _kd_term(self, pred, teacher_out, gt_clear=None):
        """使用 MSE 蒸馏"""
        with torch.no_grad():
            t_out = teacher_out.detach()
        if self.use_edge_mask and gt_clear is not None:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd = F.mse_loss(pred * edge_mask, t_out * edge_mask)
        else:
            kd = F.mse_loss(pred, t_out)
        return kd

    def _d_loss(self, d_out_fake, d_out_real):
        return F.softplus(d_out_fake).mean() + F.softplus(-d_out_real).mean()

    def _g_loss(self, d_out_fake):
        return F.softplus(-d_out_fake).mean()

    def _extract_feature(self, x):
        """提取特征用于对抗判别"""
        if hasattr(self.feat_extractor, "forward_features"):
            feat = self.feat_extractor.forward_features(x)
        elif hasattr(self.feat_extractor, "features"):
            feat = self.feat_extractor.features(x)
        elif hasattr(self.feat_extractor, "encoder"):
            feat = self.feat_extractor.encoder(x)
        else:
            feat = self.feat_extractor(x)

        # 若输出仍为 3 通道图像，则投影到 feat_dim
        if feat.ndim == 4 and feat.shape[1] == 3:
            feat = self.feat_proj(feat)
        elif feat.ndim == 4 and feat.shape[1] != self.feat_dim:
            # 自动调整通道到 feat_dim
            feat = F.adaptive_avg_pool2d(feat, (feat.shape[2], feat.shape[3]))
            conv_proj = nn.Conv2d(feat.shape[1], self.feat_dim, kernel_size=1, bias=False).to(feat.device)
            feat = conv_proj(feat)

        return feat

    # -------------------------
    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        parts = {}
        D_total = torch.tensor(0.0, device=hazy.device)

        # === 主分支 ===
        pred, teacher_out = self.forward(hazy, return_teacher=True)
        rec_loss, parts_rec = criterion(pred, gt_clear, inp_for_mask=hazy)
        parts.update({f"rec_{k}": float(v) for k, v in parts_rec.items()})

        G_total = rec_loss
        kd = self._kd_term(pred, teacher_out, gt_clear=gt_clear)
        G_total += self.lambda_kd * kd
        parts["kd"] = float(kd.item()) * self.lambda_kd

        # === 特征级对抗 ===
        enable_adv = (self.lambda_adv > 0.0)
        if enable_adv:
            with torch.no_grad():
                feat_gt = self._extract_feature(gt_clear)
            feat_pred = self._extract_feature(pred)

            d_fake = self.discriminator(feat_pred.detach())
            d_real = self.discriminator(feat_gt)
            d_loss_main = self._d_loss(d_fake, d_real)
            D_total += d_loss_main
            parts["d_main"] = float(d_loss_main.item())

            g_adv = self._g_loss(self.discriminator(feat_pred))
            G_total += self.lambda_adv * g_adv
            parts["g_adv_main"] = float(g_adv.item()) * self.lambda_adv

        # === OTF 增强分支 ===
        if self.use_onfly and (epoch is not None and should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_aug, _ = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    **self._otf_cfg
                )

            pred_aug, teacher_aug = self.forward(hazy_aug, return_teacher=True)
            rec_aug, _ = criterion(pred_aug, gt_clear, inp_for_mask=hazy_aug)
            kd_aug = self._kd_term(pred_aug, teacher_aug, gt_clear=gt_clear)
            aug_term = rec_aug + self.lambda_kd * kd_aug

            if enable_adv:
                with torch.no_grad():
                    feat_gt_aug = self._extract_feature(gt_clear)
                feat_pred_aug = self._extract_feature(pred_aug)
                d_fake_aug = self.discriminator(feat_pred_aug.detach())
                d_real_aug = self.discriminator(feat_gt_aug)
                d_loss_aug = self._d_loss(d_fake_aug, d_real_aug)
                D_total += d_loss_aug
                parts["d_otf"] = float(d_loss_aug.item())

                g_adv_aug = self._g_loss(self.discriminator(feat_pred_aug))
                aug_term += self.lambda_adv * g_adv_aug
                parts["g_adv_otf"] = float(g_adv_aug.item()) * self.lambda_adv

            G_total += self.alpha_aug * aug_term
            parts["otf_rec"] = float(rec_aug.item()) * self.alpha_aug
            parts["otf_kd"] = float(kd_aug.item()) * self.alpha_aug

        # === 最终合并返回（方案A） ===
        total = G_total + D_total
        parts["total_G"] = float(G_total.item())
        parts["total_D"] = float(D_total.item())
        parts["total"] = float(total.item())

        return total, parts


from collections import deque
from einops import rearrange

class MoCoQueue:
    def __init__(self, feat_dim=64, queue_size=512):
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue = torch.randn(queue_size, feat_dim).to(self.device)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        bs = keys.shape[0]
        if self.ptr + bs > self.queue_size:
            overflow = self.ptr + bs - self.queue_size
            self.queue[self.ptr:] = keys[:bs - overflow]
            self.queue[:overflow] = keys[bs - overflow:]
            self.ptr = overflow
        else:
            self.queue[self.ptr:self.ptr + bs] = keys
            self.ptr = (self.ptr + bs) % self.queue_size

    def get_queue(self):
        return self.queue.clone().detach()

def nt_xent_loss_moco(q, k, queue, temperature=0.07):
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    queue = F.normalize(queue, dim=1)

    pos_logits = torch.sum(q * k, dim=1, keepdim=True)  # [B, 1]
    neg_logits = torch.matmul(q, queue.T)               # [B, K]

    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)  # 正样本在第一位

    return F.cross_entropy(logits, labels)

class WaveletKDDiffusion_EM_mococontrast_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_contrast=0.5, levels=2,
                 queue_size=512, momentum=0.999,
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 # ==== OTF 参数 ====
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_contrast = lambda_contrast
        self.levels = levels
        self.use_onfly = use_onfly
        self.use_dynamic_alpha = use_dynamic_alpha
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.is_kd = True
        self.momentum = momentum

        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        # Teacher Encoder (frozen)
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Wavelet Gating
        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        # Student + Momentum Encoder + Projector
        self.encoder_q = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU())
        self.encoder_k = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU())

        # 初始化 encoder_k 的参数为 encoder_q
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Queue
        self.queue = MoCoQueue(feat_dim=64, queue_size=queue_size)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, hazy, return_feature=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta
        out = torch.clamp(x, 0, 1)

        if return_feature:
            q_feat = self.encoder_q(out)
            q_proj = self.proj_head(q_feat)

            with torch.no_grad():
                self._momentum_update_key_encoder()
                k_feat = self.encoder_k(base_out)
                k_proj = self.proj_head(k_feat)

            return out, base_out, q_proj, k_proj

        return out

    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        pred, teacher_out, feat_q, feat_k = self.forward(hazy, return_feature=True)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # MoCo Contrastive Loss
        contrast_loss = nt_xent_loss_moco(feat_q, feat_k, self.queue.get_queue())
        total_loss += self.lambda_contrast * contrast_loss
        parts["contrast"] = float(contrast_loss.item()) * self.lambda_contrast

        # Enqueue
        self.queue.enqueue_dequeue(feat_k)

        # OTF Loss
        if self.use_onfly and (epoch is None or should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(
                    gt_clear, device=gt_clear.device,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    dtype=gt_clear.dtype, **self.otf_kwargs)
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts




class WaveletKDDiffusion_EM_multiviewcontrast_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_contrast=0.5, levels=2,
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 # ==== OTF 参数 ====
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_contrast = lambda_contrast
        self.levels = levels
        self.use_onfly = use_onfly
        self.use_dynamic_alpha = use_dynamic_alpha
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.is_kd = True

        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )

        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, hazy, gt_clear=None, return_feature=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta
        out = torch.clamp(x, 0, 1)

        if return_feature:
            feat_s = self.proj_head(self.feature_extractor(out))
            with torch.no_grad():
                feat_t = self.proj_head(self.feature_extractor(base_out))
                feat_gt = self.proj_head(self.feature_extractor(gt_clear)) if gt_clear is not None else None
            return out, base_out, feat_s, feat_t, feat_gt

        return out

    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        pred, teacher_out, feat_s, feat_t, feat_gt = self.forward(hazy, gt_clear=gt_clear, return_feature=True)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === Multi-view Contrast ===
        loss_st = nt_xent_loss(feat_s, feat_t)
        loss_sg = nt_xent_loss(feat_s, feat_gt)
        loss_tg = nt_xent_loss(feat_t, feat_gt)
        contrast_loss = (loss_st + loss_sg + loss_tg) / 3.0

        total_loss += self.lambda_contrast * contrast_loss
        parts["contrast_st"] = float(loss_st.item())
        parts["contrast_sg"] = float(loss_sg.item())
        parts["contrast_tg"] = float(loss_tg.item())
        parts["contrast_total"] = float(contrast_loss.item()) * self.lambda_contrast

        if self.use_onfly and (epoch is None or should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(
                    gt_clear, device=gt_clear.device,
                    use_dynamic_alpha=self.use_dynamic_alpha,
                    dtype=gt_clear.dtype, **self.otf_kwargs)
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts




# === Patch-wise contrast loss ===
def nt_xent_loss_patch(q, k, temperature=0.5):

    
    # 检查 q 和 k 的形状是否为 5D，如果不是，进行适当的调整
    if len(q.shape) == 4:
        B, N, C, HW = q.shape  # 如果是 [B, N, C, H * W]
        pH, pW = int(HW ** 0.5), int(HW ** 0.5)  # 假设是正方形 patch
    elif len(q.shape) == 5:
        B, N, C, pH, pW = q.shape  # 如果是 [B, N, C, patch_size, patch_size]
    else:
        raise ValueError(f"Unexpected shape for q: {q.shape}")
    
    q = q.view(B, N, -1)  # 展开为 [B, N, C * pH * pW]
    k = k.view(B, N, -1)  # 展开为 [B, N, C * pH * pW]

    # 对 q 和 k 进行归一化
    q = F.normalize(q, dim=2)
    k = F.normalize(k, dim=2)

    # 使用 einsum 计算对比损失
    logits = torch.einsum("bnd,bmd->bnm", q, k) / temperature
    labels = torch.arange(N, device=q.device)

    # 计算每个样本的对比损失
    loss = sum(F.cross_entropy(logits[i], labels) for i in range(B)) / B
    return loss



# === Patch validity filter ===
def is_valid_negative(patch, anchor, brightness_thresh=0.05, var_thresh=0.01):
    return (patch.mean() - anchor.mean()).abs() > brightness_thresh and patch.var() > var_thresh

# === Model ===
class WaveletKDDiffusion_EM_patchcontrast_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_contrast=0.5, levels=2,
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5, use_dynamic_alpha=False,
                 patch_size=32, max_negatives=4,
                 otf_beta=(0.8, 1.6), otf_wp=0.8, otf_wl=0.4, otf_wv=0.2, otf_we=0.2,
                 otf_a0=(0.90, 0.88, 0.85), otf_alpha0=(0.85, 0.98),
                 otf_gamma=(0.10, 0.30), otf_glow_prob=0.5):
        super().__init__()
        self.steps = steps
        self.lambda_contrast = lambda_contrast
        self.levels = levels
        self.use_onfly = use_onfly
        self.use_dynamic_alpha = use_dynamic_alpha
        self.p_aug = p_aug
        self.alpha_aug = alpha_aug
        self.patch_size = patch_size
        self.max_negatives = max_negatives
        self.is_kd = True

        # OTF增强和参数初始化
        self.otf_kwargs = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma, glow_prob=otf_glow_prob
        )
        
        # 模型定义（教师模型加载）
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v for k, v in state_dict.items()}
            self.teacher.load_state_dict(new_state_dict, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.freq_ch = 3 * 3 * levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def extract_patches(self, feat_map):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold(feat_map)  # [B, C * patch_size * patch_size, N]
        B, _, N = patches.shape
        patches = patches.transpose(1, 2).reshape(B, N, feat_map.size(1), self.patch_size, self.patch_size)  # [B, N, C, patch_size, patch_size]

        valid_patches = []
        for b in range(B):
            valid_patch_indices = []
            for i in range(N):
                patch = patches[b, i]
                if is_valid_negative(patch, patch):  # 使用自身来做预筛选
                    valid_patch_indices.append(i)

            # 如果所有 patch 都不合适，则随机裁剪
            if len(valid_patch_indices) == 0:
                valid_patch_indices = random.sample(range(N), min(2, N))  # 随机选择几个 patch

            # Convert valid patches to Tensor (ensure it's a Tensor)
            valid_patches.append(torch.stack([patches[b, idx] for idx in valid_patch_indices]))

        return torch.stack(valid_patches)  # Make sure to return a Tensor

    def forward(self, hazy, return_patch_features=False):
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, self.levels)
            freq_base = extract_multiscale_wavelet(base_out, self.levels)
            gated_freq = self.freq_gating(hazy - base_out, freq_hazy - freq_base)

        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            x = x + self.refine_steps[i](cond)
        out = torch.clamp(x, 0, 1)

        if return_patch_features:
            feat_s = self.feature_extractor(out)
            feat_t = self.feature_extractor(base_out)
            ps, pt = self.extract_patches(feat_s), self.extract_patches(feat_t)
            
            # Ensure that ps and pt are Tensors
            B, N, C, pH, pW = ps.shape

            q, k = [], []
            for b in range(B):
                for i in range(N):
                    a_s, a_t = ps[b, i], pt[b, i]
                    q_feat = self.proj_head(a_s.unsqueeze(0))
                    k_feat_pos = self.proj_head(a_t.unsqueeze(0))
                    k_feats = [k_feat_pos]
                    fallback_used = False
                    for j in range(N):
                        if j == i: continue
                        cand = pt[b, j]
                        if is_valid_negative(cand, a_t):
                            k_feats.append(self.proj_head(cand.unsqueeze(0)))
                        if len(k_feats) >= self.max_negatives + 1:
                            break
                    if len(k_feats) == 1:
                        rand_j = random.choice([j for j in range(N) if j != i])
                        k_feats.append(self.proj_head(pt[b, rand_j].unsqueeze(0)))
                    q.append(q_feat)
                    k.append(torch.cat(k_feats, dim=0).mean(dim=0, keepdim=True))
            return out, base_out, torch.stack(q, dim=0).unsqueeze(0), torch.stack(k, dim=0).unsqueeze(0)

        return out


    def compute_loss(self, hazy, gt_clear, criterion, epoch=None):
        pred, _, feat_q, feat_k = self.forward(hazy, return_patch_features=True)
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        contrast_loss = nt_xent_loss_patch(feat_q, feat_k)
        total_loss += self.lambda_contrast * contrast_loss
        parts["contrast_patch"] = float(contrast_loss.item()) * self.lambda_contrast

        if self.use_onfly and (epoch is None or should_enable_otf(epoch)) and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(gt_clear, device=gt_clear.device,
                    use_dynamic_alpha=self.use_dynamic_alpha, dtype=gt_clear.dtype, **self.otf_kwargs)
            pred_otf = self.forward(hazy_otf)
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)
            total_loss += self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts

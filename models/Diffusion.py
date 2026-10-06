import torch
import torch.nn as nn
import torch.nn.functional as F
from .Restormer import TransformerBlock, Restormer

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


# ====== GaussianDiffusion（修复拼接逻辑） ======

# class GaussianDiffusion(nn.Module):
#     def __init__(self, model, image_size=None, channels=3, timesteps=1000):
#         super().__init__()
#         self.model = model
#         self.timesteps = timesteps

#         betas = torch.linspace(1e-4, 0.02, timesteps)
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, dim=0)

#         self.register_buffer('betas', betas)
#         self.register_buffer('alphas', alphas)
#         self.register_buffer('alphas_cumprod', alphas_cumprod)
#         self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

#     def q_sample(self, x_start, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         return (
#             extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )

#     def forward(self, hazy_aug, residual_gt, t):
#         noise = torch.randn_like(residual_gt)
#         x_noisy = self.q_sample(residual_gt, t, noise)
#         cond = torch.cat([hazy_aug, x_noisy], dim=1)
#         pred_residual = self.model(cond)
#         return F.mse_loss(pred_residual, residual_gt)

#     @torch.no_grad()
#     def sample(self, hazy_aug, steps=25):
#         B, C, H, W = hazy_aug.shape
#         device = hazy_aug.device
#         z = torch.randn(B, 3, H, W, device=device)
#         for i in reversed(range(steps)):
#             t = torch.full((B,), i, device=device, dtype=torch.long)
#             cond = torch.cat([hazy_aug, z], dim=1)
#             pred_residual = self.model(cond)
#             z = pred_residual
#         return pred_residual
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

# ====== Restormer + Diffusion（修复推理拼接） ======

# class RestormerDiffusion(nn.Module):
#     def __init__(self, base_ckpt_path=None, base_net=None, steps=1000):
#         super().__init__()
#         self.steps = steps

#         self.base_net = base_net if base_net is not None else TinyBaseNet()
#         if base_ckpt_path is not None:
#             ckpt = torch.load(base_ckpt_path, map_location='cpu')
#             state_dict = ckpt.get("state_dict", ckpt)
#             new_state_dict = {
#                 k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
#                 for k, v in state_dict.items()
#             }
#             self.base_net.load_state_dict(new_state_dict, strict=False)
#         for p in self.base_net.parameters():
#             p.requires_grad = False

#         self.time_embed = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU()
#         )

#         # 最终输入 = hazy_aug (67) + noisy (3) = 70
#         self.residual_predictor = UNetResidualPredictor(in_ch=70, base_ch=64, out_ch=3)

#         self.diffusion = GaussianDiffusion(
#             model=self.residual_predictor,
#             image_size=None,
#             channels=3,
#             timesteps=steps
#         )

#     def forward(self, hazy, gt_clear=None, t=None, training=True):
#         with torch.no_grad():
#             base_out = self.base_net(hazy)

#         if t is None:
#             t = torch.randint(0, self.diffusion.timesteps, (hazy.size(0),), device=hazy.device)

#         t_norm = t.float() / float(self.steps)
#         t_embed = self.time_embed(t_norm.unsqueeze(1))
#         t_embed_expanded = t_embed[:, :, None, None].expand(-1, -1, hazy.shape[2], hazy.shape[3])

#         cond_extra = torch.cat([base_out, t_embed_expanded], dim=1)  # 67 通道

#         if training:
#             assert gt_clear is not None
#             residual_gt = gt_clear - base_out
#             # ✅ 这里传进去的 cond 是 67 通道，diffusion 内部再拼 noisy(3) → 70
#             return self.diffusion(cond_extra, residual_gt, t)
#         else:
#             # 推理阶段同样只拼 base_out + time
#             residual_pred = self.diffusion.sample(cond_extra)  # sample 内部再拼 z(3)
#             return torch.clamp(base_out + residual_pred, 0, 1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn

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

class RestormerDiffusion(nn.Module):
    def __init__(self, restormer_model=None, base_ckpt_path=None, steps=1000):
        super().__init__()
        self.steps = steps

        # ==== Restormer 主干网络（冻结）====
        self.base_net = restormer_model if restormer_model is not None else Restormer(
            inp_channels=3, out_channels=3, dim=32,
            num_blocks=[4, 4, 4], num_refinement_blocks=2,
            heads=[1, 2, 4], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="BiasFree"
        )

        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }

            model_dict = self.base_net.state_dict()
            filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            self.base_net.load_state_dict(filtered_dict, strict=False)
            print(f"[RestormerDiffusion] Loaded partial checkpoint: {base_ckpt_path}")
            print(f"  Loaded {len(filtered_dict)} layers | Skipped {len(new_state_dict) - len(filtered_dict)}")

        for p in self.base_net.parameters():
            p.requires_grad = False

        # ==== 时间嵌入 ====
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # ==== 残差预测器（TTT优化的主体）====
        self.residual_predictor = UNetResidualPredictor(in_ch=70, base_ch=64, out_ch=3)

        # ==== 扩散模型 ====
        self.diffusion = GaussianDiffusion(
            model=self.residual_predictor,
            image_size=None,
            channels=3,
            timesteps=steps
        )

        # ==== 默认加雾强度 ====
        self.default_haze_strength = 0.2

    def forward(self, hazy, gt_clear=None, t=None, training=None):
        """
        如果 training=None，将使用 self.training 控制（更稳）
        """
        base_out = self.base_net(hazy)
        if t is None:
            t = torch.randint(0, self.diffusion.timesteps, (hazy.size(0),), device=hazy.device)

        t_norm = t.float() / float(self.steps)
        t_embed = self.time_embed(t_norm.unsqueeze(1))
        t_embed_expanded = t_embed[:, :, None, None].expand(-1, -1, hazy.shape[2], hazy.shape[3])
        cond_extra = torch.cat([base_out, t_embed_expanded], dim=1)

        # fallback to model mode
        if training is None:
            training = self.training

        if training:
            residual_pred = self.diffusion.predict(cond_extra, t)
            pred = torch.clamp(base_out + residual_pred, 0, 1)
            return pred
        else:
            residual_pred = self.diffusion.sample(cond_extra)
            return torch.clamp(base_out + residual_pred, 0, 1)

    def add_haze(self, x, strength=0.2, A=1.0):
        """模拟加雾过程：x*(1-a) + A*a"""
        return x * (1 - strength) + A * strength

    def get_ttt_params(self):
        """返回需要 TTT 更新的参数"""
        return list(self.residual_predictor.parameters())

    def compute_ttt_loss(self, x_r, step_idx=0, total_steps=1,
                         lambda_charb=0.6, lambda_ssim=0.3, lambda_tv=0.1):
        """
        Test-Time Training 自监督损失（无 GT，仅伪监督）
        可在主程序中使用 EMA、早停等策略
        """
        self.train()
        x_r_haze2 = self.add_haze(x_r, strength=self.default_haze_strength)
        pred = self.forward(x_r_haze2, training=True)

        loss_charb = charbonnier_loss(pred, x_r)
        loss_ssim  = ssim_loss(pred, x_r)
        loss_tv    = total_variation_loss(pred)

        loss = lambda_charb * loss_charb + lambda_ssim * loss_ssim + lambda_tv * loss_tv
        return loss

# ====== Pure Diffusion ======
class PureDiffusion(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        self.denoiser = UNetResidualPredictor(in_ch=6, base_ch=64, out_ch=3)
        self.diffusion = GaussianDiffusion(self.denoiser, image_size=None, channels=3, timesteps=steps)

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        if training:
            assert gt_clear is not None and t is not None
            residual_gt = gt_clear - hazy
            return self.diffusion(hazy, residual_gt, t)
        else:
            residual_pred = self.diffusion.sample(hazy)
            return hazy + residual_pred


# ====== BaseNet + Diffusion ======
class BaseCondDiffusion(nn.Module):
    def __init__(self, diffusion_steps=1000):
        super().__init__()
        self.base_net = TinyBaseNet()
        self.denoiser = UNetResidualPredictor(in_ch=6, base_ch=64, out_ch=3)
        self.diffusion = GaussianDiffusion(self.denoiser, image_size=None, channels=3, timesteps=diffusion_steps)

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        base_out = self.base_net(hazy)
        if training:
            assert gt_clear is not None and t is not None
            residual_gt = gt_clear - base_out
            return self.diffusion(hazy, residual_gt, t)
        else:
            residual_pred = self.diffusion.sample(hazy)
            return base_out + residual_pred


from .nafnet import NAFNet

class NAFNetDiffusion(nn.Module):
    def __init__(self, diffusion_steps=500):
        super().__init__()
        self.unet = NAFNet(
            img_channel=6,
            width=32,
            middle_blk_num=4,
            enc_blk_nums=[1, 1, 2, 4],
            dec_blk_nums=[1, 1, 1, 1]
        )
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=None,
            channels=3,
            timesteps=diffusion_steps
        )

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        if training:
            assert gt_clear is not None and t is not None
            residual = gt_clear - hazy
            return self.diffusion(hazy, residual, t)
        else:
            residual_pred = self.diffusion.sample(hazy)
            return hazy + residual_pred

from .Uformer import Uformer

class UformerDiffusion(nn.Module):
    def __init__(self, diffusion_steps=500):
        super().__init__()
        self.unet = Uformer(in_ch=6, out_ch=3, base_dim=32)
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=None,
            channels=3,
            timesteps=diffusion_steps
        )

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        if training:
            assert gt_clear is not None and t is not None
            residual = gt_clear - hazy
            return self.diffusion(hazy, residual, t)
        else:
            residual_pred = self.diffusion.sample(hazy)
            return hazy + residual_pred


from .UNet_CBAM import UNet_CBAM

class CBAMDiffusion(nn.Module):
    def __init__(self, diffusion_steps=500):
        super().__init__()
        self.unet = UNet_CBAM(in_ch=6, base_ch=64, out_ch=3)
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=None,
            channels=3,
            timesteps=diffusion_steps
        )

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        if training:
            assert gt_clear is not None and t is not None
            residual = gt_clear - hazy
            return self.diffusion(hazy, residual, t)
        else:
            residual_pred = self.diffusion.sample(hazy)
            return hazy + residual_pred
        



class OneStepDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        # 冻结 base net
        self.base_net = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("net.") or k.startswith("model."):
                    k = ".".join(k.split(".")[1:])
                new_state_dict[k] = v
            self.base_net.load_state_dict(new_state_dict, strict=False)
        for p in self.base_net.parameters():
            p.requires_grad = False

        # 添加时间嵌入模块（简单的MLP）
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # UNet 预测残差 + 时间嵌入融合（通过通道重复）
        self.residual_predictor = UNetResidualPredictor(in_ch=6 + 64, base_ch=64, out_ch=3)

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            base_out = self.base_net(hazy)

        cond = torch.cat([hazy, base_out], dim=1)  # 原图 + base

        if training:
            assert gt_clear is not None and t is not None
            residual_gt = gt_clear - base_out

            # 生成时间嵌入并扩展为图像大小
            t_norm = t.float() / 1000.0
            t_embed = self.time_embed(t_norm.unsqueeze(1))
            t_embed_expanded = t_embed[:, :, None, None].expand(-1, -1, hazy.shape[2], hazy.shape[3])

            cond_with_time = torch.cat([cond, t_embed_expanded], dim=1)
            residual_pred = self.residual_predictor(cond_with_time)
            return F.mse_loss(residual_pred, residual_gt)

        else:
            # 推理时 t 默认为一个中间值（例如 t=500）
            t = torch.full((hazy.size(0),), 500, device=hazy.device, dtype=torch.long)
            t_norm = t.float() / 1000.0
            t_embed = self.time_embed(t_norm.unsqueeze(1))
            t_embed_expanded = t_embed[:, :, None, None].expand(-1, -1, hazy.shape[2], hazy.shape[3])
            cond_with_time = torch.cat([cond, t_embed_expanded], dim=1)

            residual_pred = self.residual_predictor(cond_with_time)
            return torch.clamp(base_out + residual_pred, 0, 1)



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


class MultiStepTransformerDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1):
        super().__init__()
        assert 1 <= steps <= 10, "steps must be between 1 and 10"
        self.steps = steps

        self.base_net = base_net if base_net is not None else TinyBaseNet()

        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("net.") or k.startswith("model."):
                    k = ".".join(k.split(".")[1:])
                new_state_dict[k] = v

            missing_keys, unexpected_keys = self.base_net.load_state_dict(new_state_dict, strict=False)
            if missing_keys or unexpected_keys:
                print(f"[MultiStepTransformerDiffusion] Loaded base_ckpt with:")
                if missing_keys:
                    print(f"  - Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"  - Unexpected keys: {unexpected_keys}")
            else:
                print("[MultiStepTransformerDiffusion] Successfully loaded base_ckpt.")

        for p in self.base_net.parameters():
            p.requires_grad = False

        # 每一步都使用不同的 RestormerRefineNet（不共享）
        self.refine_steps = nn.ModuleList([
            RestormerRefineNet(
                in_ch=6,
                base_ch=32,
                num_blocks=1
            ) for _ in range(self.steps)
        ])

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            x = self.base_net(hazy)

        for i in range(self.steps):
            cond = torch.cat([hazy, x], dim=1)
            residual = self.refine_steps[i](cond)
            x = x + residual

        if training and gt_clear is not None:
            return F.mse_loss(x, gt_clear)
        else:
            return x


# ====== Conditional Feature Fusion (CFF) Block ======
class CFFBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.encoder_pre = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
        self.encoder_in = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
        self.weight = nn.Conv2d(2 * ch, ch, 1)
        self.bias = nn.Conv2d(2 * ch, ch, 1)
        self.out = nn.Conv2d(ch, 6, 3, 1, 1)  # 新增：输出 6 通道

    def forward(self, xpre, xinput):
        Fp = self.encoder_pre(xpre)
        Fin = self.encoder_in(xinput)
        Fc = torch.cat([Fp, Fin], dim=1)
        omega = self.weight(Fc)
        gamma = self.bias(Fc)
        fused = omega * Fp + gamma
        return self.out(fused)  # 融合后再输出 6 通道


# ====== CCFDiffusion（支持多步扩散 + CFF 感知解码） ======
class CCFDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, ch=64, steps=1):
        super().__init__()
        assert 1 <= steps <= 10, "steps must be between 1 and 10"
        self.steps = steps

        self.base_net = base_net if base_net is not None else TinyBaseNet()

        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("net.") or k.startswith("model."):
                    k = ".".join(k.split(".")[1:])
                new_state_dict[k] = v
            self.base_net.load_state_dict(new_state_dict, strict=False)

        for p in self.base_net.parameters():
            p.requires_grad = False

        self.cff = CFFBlock(ch=3)

        self.refine_step = nn.ModuleList([
            UNetResidualPredictor(in_ch=6 + 64, base_ch=64, out_ch=3)
            for _ in range(self.steps)
        ])

        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            xpre = self.base_net(hazy)

        cond = self.cff(xpre, hazy)

        if t is None:
            t = torch.full((hazy.size(0),), 500, device=hazy.device, dtype=torch.long)

        t_norm = t.float() / 1000.0
        t_embed = self.time_embed(t_norm.unsqueeze(1))
        t_embed_expanded = t_embed[:, :, None, None].expand(-1, -1, hazy.shape[2], hazy.shape[3])

        x = xpre
        for step in self.refine_step:
            step_input = torch.cat([cond, t_embed_expanded], dim=1)
            residual = step(step_input)
            x = x + residual

        if training and gt_clear is not None:
            return F.l1_loss(x, gt_clear) + 0.5 * F.l1_loss(x * 255., gt_clear * 255.)
        else:
            return torch.clamp(x, 0, 1)



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

# ====== Wavelet Diffusion with Multiscale Gated Frequency Guidance ======
class WaveletDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1, levels=2):
        super().__init__()
        assert 1 <= steps <= 10, "steps must be between 1 and 10"
        self.steps = steps
        self.levels = levels

        self.base_net = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("net.") or k.startswith("model."):
                    k = ".".join(k.split(".")[1:])
                new_state_dict[k] = v
            self.base_net.load_state_dict(new_state_dict, strict=False)

        for p in self.base_net.parameters():
            p.requires_grad = False

        # freq_ch = C × 3 × levels, 对 RGB 和 2层小波
        freq_ch = 3 * 3 * self.levels
        self.refine_steps = nn.ModuleList([
            WRRefineNet(in_ch=6 + freq_ch) for _ in range(self.steps)
        ])
        self.gating_module = ResidualFrequencyGating(img_ch=3, freq_ch=freq_ch)

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            base_out = self.base_net(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            residual = hazy - base_out
            freq_input = freq_hazy - freq_base  # 重点保留纹理差异部分
            gated_freq = self.gating_module(residual, freq_input)

        x = base_out
        for i in range(self.steps):
            cond = torch.cat([hazy, x, gated_freq], dim=1)  # 3 + 3 + freq_ch
            residual = self.refine_steps[i](cond)
            x = x + residual

        if training and gt_clear is not None:
            return F.mse_loss(x, gt_clear)
        else:
            return torch.clamp(x, 0, 1)


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

# class WaveletBidirectionalDiffusion(nn.Module):
#     def __init__(self, base_ckpt_path=None, base_net=None, steps=1, levels=2, reverse_weight=0.05):
#         super().__init__()
#         assert 1 <= steps <= 10
#         self.steps = steps
#         self.levels = levels
#         self.reverse_weight = reverse_weight

#         self.base_net = base_net if base_net is not None else TinyBaseNet()
#         if base_ckpt_path is not None:
#             ckpt = torch.load(base_ckpt_path, map_location='cpu')
#             state_dict = ckpt.get("state_dict", ckpt)
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith("module."):
#                     k = k[len("module."):]
#                 if k.startswith("net.") or k.startswith("model."):
#                     k = ".".join(k.split(".")[1:])
#                 new_state_dict[k] = v
#             self.base_net.load_state_dict(new_state_dict, strict=False)

#         for p in self.base_net.parameters():
#             p.requires_grad = False

#         freq_ch = 3 * 3 * self.levels
#         self.gating_module = ResidualFrequencyGating(img_ch=3, freq_ch=freq_ch)
#         self.refine_steps = nn.ModuleList([
#             WRRefineNet(in_ch=6 + freq_ch) for _ in range(self.steps)
#         ])
#         self.reverse_net = ReverseNet(in_ch=3)  # 反向网络 base_out → hazy

#     def forward(self, hazy, gt_clear=None, t=None, training=True):
#         with torch.no_grad():
#             base_out = self.base_net(hazy)
#             freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
#             freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
#             residual = hazy - base_out
#             freq_input = freq_hazy - freq_base
#             gated_freq = self.gating_module(residual, freq_input)

#         x = base_out
#         for i in range(self.steps):
#             cond = torch.cat([hazy, x, gated_freq], dim=1)
#             res = self.refine_steps[i](cond)
#             x = x + res

#         if training and gt_clear is not None:
#             loss_forward = F.mse_loss(x, gt_clear)
#             recon_hazy = self.reverse_net(base_out.detach())  # 🔁 反向预测 haze
#             loss_reverse = F.mse_loss(recon_hazy, hazy)
#             return loss_forward + self.reverse_weight * loss_reverse
#         else:
#             return torch.clamp(x, 0, 1)

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


# ==== 主结构 ====
# class WaveletBidirectionalDiffusion(nn.Module):
#     def __init__(self, base_ckpt_path=None, base_net=None,
#                  steps=1, levels=2, reverse_weight=0.05, use_perceptual=False):
#         super().__init__()
#         assert 1 <= steps <= 10
#         self.steps = steps
#         self.levels = levels
#         self.reverse_weight = reverse_weight
#         self.use_perceptual = use_perceptual

#         self.base_net = base_net if base_net is not None else TinyBaseNet()
#         if base_ckpt_path is not None:
#             ckpt = torch.load(base_ckpt_path, map_location='cpu')
#             state_dict = ckpt.get("state_dict", ckpt)
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith("module."):
#                     k = k[len("module."):]
#                 if k.startswith("net.") or k.startswith("model."):
#                     k = ".".join(k.split(".")[1:])
#                 new_state_dict[k] = v
#             self.base_net.load_state_dict(new_state_dict, strict=False)

#         for p in self.base_net.parameters():
#             p.requires_grad = False

#         freq_ch = 3 * 3 * self.levels
#         self.gating_module = ResidualFrequencyGating(img_ch=3, freq_ch=freq_ch)
#         self.refine_steps = nn.ModuleList([
#             WRRefineNet(in_ch=6 + freq_ch) for _ in range(self.steps)
#         ])
#         self.reverse_net = ReverseNet(in_ch=3)
#         self.perceptual_loss = PerceptualLoss() if self.use_perceptual else None

#     def forward(self, hazy, gt_clear=None, t=None, training=True):
#         with torch.no_grad():
#             base_out = self.base_net(hazy)
#             freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
#             freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
#             residual = hazy - base_out
#             freq_input = freq_hazy - freq_base
#             gated_freq = self.gating_module(residual, freq_input)

#         x = base_out
#         for i in range(self.steps):
#             cond = torch.cat([hazy, x, gated_freq], dim=1)
#             delta = self.refine_steps[i](cond)
#             x = x + delta

#         if training and gt_clear is not None:
#             loss_forward = F.mse_loss(x, gt_clear)

#             with torch.no_grad():
#                 edge_mask = compute_edge_mask(hazy)  # [B,1,H,W]
#                 edge_mask = edge_mask.expand_as(hazy)

#             pred_hazy = self.reverse_net(base_out.detach())
#             if self.perceptual_loss:
#                 loss_reverse = self.perceptual_loss(pred_hazy * edge_mask, hazy * edge_mask)
#             else:
#                 loss_reverse = F.mse_loss(pred_hazy * edge_mask, hazy * edge_mask)

#             return loss_forward + self.reverse_weight * loss_reverse
#         else:
#             return torch.clamp(x, 0, 1)

class WaveletBidirectionalDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, levels=2, reverse_weight=0.05, use_perceptual=False):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.levels = levels
        self.reverse_weight = reverse_weight
        self.use_perceptual = use_perceptual

        # ---- 冻结 base_net ----
        self.base_net = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module.") :]
                if k.startswith("net.") or k.startswith("model."):
                    k = ".".join(k.split(".")[1:])
                new_state_dict[k] = v
            self.base_net.load_state_dict(new_state_dict, strict=False)

        for p in self.base_net.parameters():
            p.requires_grad = False

        # ---- 频域 gating + refine ----
        freq_ch = 3 * 3 * self.levels
        self.gating_module = ResidualFrequencyGating(img_ch=3, freq_ch=freq_ch)
        self.refine_steps = nn.ModuleList(
            [WRRefineNet(in_ch=6 + freq_ch) for _ in range(self.steps)]
        )

        # ---- reverse net ----
        self.reverse_net = ReverseNet(in_ch=3)
        self.perceptual_loss = PerceptualLoss() if self.use_perceptual else None

    def forward(self, hazy, return_feats=False):
        """仅做前向推理"""
        with torch.no_grad():
            base_out = self.base_net(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            residual = hazy - base_out
            freq_input = freq_hazy - freq_base
            gated_freq = self.gating_module(residual, freq_input)

        x = base_out
        for i in range(self.steps):
            cond = torch.cat([hazy, x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_feats:
            return out, base_out  # base_out 作为 encoder 特征
        return out

    def compute_loss(self, hazy, gt_clear, criterion):
        """训练时调用，使用 CombinedLoss + reverse loss"""
        pred, feat_inp = self.forward(hazy, return_feats=True)
        feat_tar = self.base_net(gt_clear).detach()

        # === 1. CombinedLoss ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)
        # === 2. Reverse loss ===
        with torch.no_grad():
            edge_mask = compute_edge_mask(hazy).expand_as(hazy)

        pred_hazy = self.reverse_net(feat_inp.detach())
        if self.perceptual_loss:
            loss_reverse = self.perceptual_loss(pred_hazy * edge_mask, hazy * edge_mask)
        else:
            loss_reverse = F.mse_loss(pred_hazy * edge_mask, hazy * edge_mask)

        total_loss = total_loss + self.reverse_weight * loss_reverse
        parts["reverse"] = float(loss_reverse.item()) * self.reverse_weight

        return total_loss, parts


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


# ========= 主模型（修改版） =========
class WaveletRDiffusion(nn.Module):
    """
    含反向监督的条件式 Diffusion 模型
    """
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, levels=2, use_perceptual=False,
                 gamma=1.0, rel_w=0.2, rel_margin=0.0,
                 step_max_scale=2.0, reverse_weight=0.05):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.levels = levels
        self.use_perceptual = use_perceptual
        self.gamma = float(gamma)
        self.rel_w = float(rel_w)
        self.rel_margin = float(rel_margin)
        self.step_max_scale = float(step_max_scale)
        self.reverse_weight = float(reverse_weight)

        # ========== Base Network ==========
        self.base_net = base_net if base_net is not None else nn.Identity()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."): k = k[len("module."):]
                if k.startswith("net.") or k.startswith("model."):
                    k = ".".join(k.split(".")[1:])
                new_state_dict[k] = v
            self.base_net.load_state_dict(new_state_dict, strict=False)

        for p in self.base_net.parameters():
            p.requires_grad = False

        # ========== 模块初始化 ==========
        freq_ch = 3 * 3 * self.levels
        self.gating_module = ResidualFrequencyGating(img_ch=3, freq_ch=freq_ch)
        self.blender = StartBlender(in_ch=6)
        self._cond_ch = 3 + 3 + freq_ch + 3 + 1 + 1  # hazy + x + gated_freq + residual + alpha + conf_mean

        self.refine_steps = nn.ModuleList([
            WRRefineNet(in_ch=self._cond_ch) for _ in range(self.steps)
        ])
        self.step_scalers = nn.ModuleList([
            StepScaler(in_ch=self._cond_ch, max_scale=self.step_max_scale) for _ in range(self.steps)
        ])

        self.perceptual_loss = PerceptualLoss() if self.use_perceptual else None
        self.reverse_net = ReverseNet(in_ch=3)

    def _build_freq_conf(self, freq_hazy, freq_base):
        df = (freq_hazy - freq_base).abs()
        conf = torch.exp(-self.gamma * df).clamp(min=0.0, max=1.0)
        conf_mean = conf.mean(dim=1, keepdim=True)
        freq_in = conf * (freq_hazy - freq_base) + (1.0 - conf) * freq_hazy
        return freq_in, conf_mean

    def forward(self, hazy, gt_clear=None, training=True):
        # === Step 1: Base output & frequency ===
        with torch.no_grad():
            base_out = self.base_net(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)

        # === Step 2: Learnable fusion as x0 ===
        x0, alpha = self.blender(hazy, base_out)
        residual0 = hazy - x0

        # === Step 3: Frequency confidence gating ===
        freq_in, conf_mean = self._build_freq_conf(freq_hazy, freq_base)
        gated_freq = self.gating_module(residual0, freq_in)

        # === Step 4: Iterative refine with adaptive step ===
        x = x0
        for i in range(self.steps):
            cond = torch.cat([hazy, x, gated_freq, residual0, alpha, conf_mean], dim=1)
            delta = self.refine_steps[i](cond)
            s = self.step_scalers[i](cond)
            x = x + s * delta

        # === Step 5: Loss ===
        if training and gt_clear is not None:
            loss_main = self.perceptual_loss(x, gt_clear) if self.perceptual_loss else F.mse_loss(x, gt_clear)
            loss_rel = relative_improve_loss(x, base_out.detach(), gt_clear, margin=self.rel_margin, p=1)

            hazy_hat = self.reverse_net(x)
            loss_reverse = F.l1_loss(hazy_hat, hazy)

            return loss_main + self.rel_w * loss_rel + self.reverse_weight * loss_reverse
        else:
            return torch.clamp(x, 0, 1)
        

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



# hazy → gt base作为引导
# class GuidedDiffusion(nn.Module):
#     def __init__(self, base_ckpt_path=None, base_net=None, steps=1, guide_mode='concat', lambda_guide=0.5):
#         super().__init__()
#         assert 1 <= steps <= 10
#         assert guide_mode in ['concat', 'attention'], "Unsupported guide_mode"

#         self.steps = steps
#         self.guide_mode = guide_mode
#         self.lambda_guide = lambda_guide
#         self.is_dual_supervision = False  # 不再 dual loss，而是 guidance-based supervision

#         # 冻结 base_net，只用作 feature guidance
#         self.base_net = base_net if base_net is not None else TinyBaseNet()
#         if base_ckpt_path is not None:
#             ckpt = torch.load(base_ckpt_path, map_location='cpu')
#             state_dict = ckpt.get("state_dict", ckpt)
#             new_state_dict = {
#                 k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
#                 for k, v in state_dict.items()
#             }
#             self.base_net.load_state_dict(new_state_dict, strict=False)
#         for p in self.base_net.parameters():
#             p.requires_grad = False

#         # 默认将 base_out 作为一个引导通道拼接输入
#         in_ch = 6 if guide_mode == 'concat' else 3
#         self.refine_steps = nn.ModuleList([
#             # UNetResidualPredictor(in_ch=in_ch) for _ in range(self.steps)
#             MiniRestormerRefineNet(in_ch=in_ch, out_ch=3, dim=32, num_levels=1)
#             for _ in range(self.steps)
#         ])

#     def forward(self, hazy, gt_clear=None, t=None, training=True):
#         with torch.no_grad():
#             base_out = self.base_net(hazy)  # [B, 3, H, W]

#         # 初始预测 = 从 hazy 直接开始
#         x = hazy.clone()
#         for i in range(self.steps):
#             if self.guide_mode == 'concat':
#                 cond = torch.cat([x, base_out], dim=1)  # [B,6,H,W] ← 仅作为引导
#             else:
#                 cond = x  # 你也可以扩展 attention 融合 base_out

#             delta = self.refine_steps[i](cond)
#             x = x + delta

#         if training and gt_clear is not None:
#             loss = F.mse_loss(x, gt_clear)
#             return loss
#         else:
#             return torch.clamp(x, 0, 1)

class GuidedDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, guide_mode='concat', lambda_guide=0.5):
        super().__init__()
        assert 1 <= steps <= 10
        assert guide_mode in ['concat', 'attention'], "Unsupported guide_mode"

        self.steps = steps
        self.guide_mode = guide_mode
        self.lambda_guide = lambda_guide

        # ---- 冻结 base_net ----
        self.base_net = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.base_net.load_state_dict(new_state_dict, strict=False)
        for p in self.base_net.parameters():
            p.requires_grad = False

        # ---- refine predictor ----
        in_ch = 6 if guide_mode == 'concat' else 3
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=in_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_base=False):
        """仅做推理，不算loss"""
        with torch.no_grad():
            base_out = self.base_net(hazy)  # teacher guidance

        x = hazy.clone()
        for i in range(self.steps):
            if self.guide_mode == 'concat':
                cond = torch.cat([x, base_out], dim=1)  # 拼接 guidance
            else:
                cond = x  # 未来可扩展 attention 模式
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_base:
            return out, base_out
        return out

    def compute_loss(self, hazy, gt_clear, criterion, use_edge_mask=False):
        """
        hazy      : [B,3,H,W]
        gt_clear  : [B,3,H,W]
        criterion : CombinedLoss 实例
        """
        pred, base_out = self.forward(hazy, return_base=True)

        # === 1. CombinedLoss ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. Edge-guided额外约束（可选） ===
        if use_edge_mask:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            edge_loss = F.l1_loss(pred * edge_mask, gt_clear * edge_mask)
            total_loss = total_loss + 0.1 * edge_loss
            parts["edge"] = float(edge_loss.item()) * 0.1

        return total_loss, parts


# hazy → gt + base作为引导 + 频域门控
class WaveletGuidedDiffusion(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1,
                 guide_mode='concat', lambda_guide=0.5, levels=2):
        super().__init__()
        assert 1 <= steps <= 10
        assert guide_mode in ['concat', 'attention'], "Unsupported guide_mode"

        self.steps = steps
        self.guide_mode = guide_mode
        self.lambda_guide = lambda_guide
        self.levels = levels
        self.is_dual_supervision = False

        # ===== Base Network 冻结加载 =====
        self.base_net = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                for k, v in state_dict.items()
            }
            self.base_net.load_state_dict(new_state_dict, strict=False)
        for p in self.base_net.parameters():
            p.requires_grad = False

        # ===== Frequency Gating 模块 =====
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH, HL, HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # ===== 引导方式：拼接 or 注意力 =====
        # if guide_mode == 'concat':
        #     in_ch = 3 + 3 + freq_ch  # x + base_out + gated_freq
        # else:
        #     in_ch = 3  # 暂未实现 attention

        # self.refine_steps = nn.ModuleList([
        #     UNetResidualPredictor(in_ch=in_ch) for _ in range(self.steps)
        # ])
        # ===== 引导方式：拼接 or 注意力 =====
        if guide_mode == 'concat':
            in_ch = 3 + 3 + self.freq_ch  # x + base_out + gated_freq
        else:
            in_ch = 3  # 暂未实现 attention

        # ===== 使用 WRRefineNet 替代 UNetResidualPredictor =====
        self.refine_steps = nn.ModuleList([
            # WRRefineNet(in_ch=in_ch) for _ in range(self.steps)
            MiniRestormerRefineNet(in_ch=3 + 3+ self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, gt_clear=None, t=None, training=True):
        with torch.no_grad():
            base_out = self.base_net(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)

        x = hazy.clone()
        for i in range(self.steps):
            if self.guide_mode == 'concat':
                cond = torch.cat([x, base_out, gated_freq], dim=1)
            else:
                cond = x  # 可以扩展 attention 模式

            delta = self.refine_steps[i](cond)
            x = x + delta

        if training and gt_clear is not None:
            loss = F.mse_loss(x, gt_clear)
            return loss
        else:
            return torch.clamp(x, 0, 1)
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
    def compute_loss(self, hazy, gt_clear, criterion,kd_criterion=None):
        """
        criterion: CombinedLoss 实例（用于 GT）
        KD 使用 MSE（无需 kd_criterion）
        """
        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        # === 1. 主任务损失（对 GT）===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. Step-wise KD (纯 MSE) ===
        kd_total = 0.0
        for step_out in step_outputs:
            if self.use_edge_mask:
                edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
                kd_total += F.mse_loss(step_out * edge_mask, teacher_out.detach() * edge_mask)
            else:
                kd_total += F.mse_loss(step_out, teacher_out.detach())
        kd_total /= self.steps

        # === 3. 总损失 ===
        total_kd_loss = self.lambda_kd * kd_total
        total_loss += total_kd_loss

        # === 4. 记录各项 ===
        parts["kd_stepwise_mse"] = float(kd_total.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())

        return total_loss, parts
# combine loss
    # def compute_loss(self, hazy, gt_clear, criterion, kd_criterion=None):
    #     """
    #     criterion: CombinedLoss 实例（用于 GT）
    #     kd_criterion: CombinedLoss 实例（用于 KD）；若为空则 fallback 到 MSE
    #     """
    #     pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

    #     # === 1. 主任务损失 ===
    #     total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

    #     # === 2. Step-wise KD (仅输出对齐) ===
    #     kd_total = 0.0
    #     kd_parts_accum = {}

    #     for step_out in step_outputs:
    #         if kd_criterion is not None:
    #             step_kd_loss, kd_parts = kd_criterion(step_out, teacher_out, inp_for_mask=hazy)
    #             kd_total += step_kd_loss
    #             for k, v in kd_parts.items():
    #                 kd_parts_accum[k] = kd_parts_accum.get(k, 0.0) + float(v.item())
    #         else:
    #             if self.use_edge_mask:
    #                 edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
    #                 kd_total += F.mse_loss(step_out * edge_mask, teacher_out.detach() * edge_mask)
    #             else:
    #                 kd_total += F.mse_loss(step_out, teacher_out.detach())

    #     kd_total /= self.steps
    #     total_kd_loss = self.lambda_kd * kd_total
    #     total_loss += total_kd_loss

    #     # === 3. 记录损失 ===
    #     if kd_criterion is not None:
    #         for k, v in kd_parts_accum.items():
    #             parts[f"kd_{k}"] = (v / self.steps) * self.lambda_kd

    #     parts["kd_stepwise"] = float(kd_total.item()) * self.lambda_kd
    #     parts["kd"] = float(total_kd_loss.item())

    #     return total_loss, parts
    
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

@torch.no_grad()
def add_smoke_endoscopic_onfly(J, *,
                               beta=(0.8, 1.6),
                               wp=0.8, wl=0.4, wv=0.2, we=0.2,
                               a0=(0.90, 0.88, 0.85),
                               alpha0=(0.85, 0.98),
                               gamma=(0.10, 0.30),
                               glow_prob=0.5,
                               device=None,
                               dtype=torch.float32):
    """
    内窥镜友好的 on-the-fly 加雾
    J: 清晰图 [B,3,H,W], in [0,1]
    返回: I(带烟), 以及中间场(便于调参/可选可视化)
    """
    if device is None: device = J.device
    J = J.to(device=device, dtype=dtype)
    B, C, H, W = J.shape

    # 共享场（广播到 batch）
    P = _perlin_fbm_like(H, W, octaves=random.choice([2,3,4]),
                         base_scale=random.choice([32,48,64]),
                         device=device, dtype=dtype)               # [1,1,H,W]
    L = _radial_light(H, W, r_ratio=random.choice([0.5,0.6,0.7]),
                      device=device, dtype=dtype)                  # [1,1,H,W]
    V = _vignette(H, W, power=random.choice([1.5,2.0,2.5]),
                  device=device, dtype=dtype)                      # [1,1,H,W]
    E = _edge_suppression(J)                                       # [B,1,H,W]

    D = wp * P + wl * L + wv * V - we * E
    D = D.clamp(0, 1)                                               # [B,1,H,W]（P/L/V 广播）

    beta_val  = torch.empty(B,1,1,1, device=device, dtype=dtype).uniform_(*beta)
    t = torch.exp(-beta_val * D)                                    # 透射率 [B,1,H,W]

    alpha0_val = torch.empty(B,1,1,1, device=device, dtype=dtype).uniform_(*alpha0)
    gamma_val  = torch.empty(B,1,1,1, device=device, dtype=dtype).uniform_(*gamma)
    a0_vec = torch.tensor(a0, device=device, dtype=dtype).view(1,3,1,1)

    # A(x)：偏灰暖，并随 L(x) 增亮
    A = alpha0_val * a0_vec * (1 + gamma_val * L)                  # [B,3,H,W] via broadcast
    A = A.clamp(0, 1)

    I = J * t + A * (1 - t)                                        # 基本散射合成

    # 可选：辉光项（强光弥散）
    if random.random() < glow_prob:
        tau   = random.uniform(0.7, 0.85)
        kappa = random.uniform(0.05, 0.12)
        bright = (J - tau).clamp(min=0.0)
        sigma_map = 1 + 5 * (1 - t)                                # [B,1,H,W]
        # 用均值模糊近似逐步扩散
        G = 0
        for _ in range(2):
            blur = F.avg_pool2d(bright, kernel_size=5, stride=1, padding=2)
            G = G + blur * sigma_map
        I = (I + kappa * G).clamp(0, 1)

    return I, {"t": t, "D": D, "L": L, "P": P, "A": A}

# =========================
# WaveletKDDiffusion_EM 的 OTF 版本
# =========================

class WaveletKDDiffusion_EM_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
                 # --- OTF 参数 ---
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5,
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
        self._otf_cfg = dict(
            beta=otf_beta, wp=otf_wp, wl=otf_wl, wv=otf_wv, we=otf_we,
            a0=otf_a0, alpha0=otf_alpha0, gamma=otf_gamma,
            glow_prob=otf_glow_prob
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

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    def forward(self, hazy, return_teacher=False):
        # === 教师分支（无梯度） ===
        with torch.no_grad():
            base_out = self.teacher(hazy)
            freq_hazy = extract_multiscale_wavelet(hazy, levels=self.levels)
            freq_base = extract_multiscale_wavelet(base_out, levels=self.levels)
            freq_residual = freq_hazy - freq_base
            residual = hazy - base_out
            gated_freq = self.freq_gating(residual, freq_residual)  # [B,freq_ch,H,W]

        # === 学生 refine ===
        x = hazy
        for i in range(self.steps):
            cond = torch.cat([x, gated_freq], dim=1)
            delta = self.refine_steps[i](cond)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, base_out
        return out

    def _kd_term(self, pred, teacher_out, gt_clear=None):
        """可选 edge-aware KD"""
        if self.use_edge_mask and gt_clear is not None:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
        else:
            kd = F.mse_loss(pred, teacher_out.detach())
        return kd

    def compute_loss(self, hazy, gt_clear, criterion):
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

        # --- 增强分支（随机触发） ---
        if self.use_onfly and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_aug, _meta = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
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

    def compute_loss(self, hazy, gt_clear, criterion,kd_criterion=None):
        """
        criterion: CombinedLoss 实例
        """
        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        # === 1. GT loss ===
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # === 2. Step-wise KD ===
        kd_step_loss = 0
        for step_out in step_outputs:
            if self.use_edge_mask:
                edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
                kd_step_loss += F.mse_loss(step_out * edge_mask, teacher_out.detach() * edge_mask)
            else:
                kd_step_loss += F.mse_loss(step_out, teacher_out.detach())
        kd_step_loss /= self.steps  # 求平均

        # === 3. 总损失 ===
        total_kd_loss = self.lambda_kd * kd_step_loss
        total_loss += total_kd_loss

        parts["kd_stepwise"] = float(kd_step_loss.item()) * self.lambda_kd
        parts["kd"] = float(total_kd_loss.item())

        return total_loss, parts
    # Combinedloss 
    # def compute_loss(self, hazy, gt_clear, criterion, kd_criterion=None):
    #     """
    #     criterion     : CombinedLoss 实例（对 GT）
    #     kd_criterion  : CombinedLoss 实例（对 teacher 蒸馏）
    #     """
    #     assert kd_criterion is not None, "Please provide kd_criterion as a CombinedLoss instance."

    #     pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

    #     # === 1. GT loss ===
    #     total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

    #     # === 2. Step-wise KD (感知增强) ===
    #     kd_total = 0
    #     kd_parts_accumulate = {}  # 用于每项 loss 求平均

    #     for step_out in step_outputs:
    #         step_loss, step_parts = kd_criterion(step_out, teacher_out, inp_for_mask=hazy)
    #         kd_total += step_loss

    #         # 汇总每一项 loss 的值
    #         for k, v in step_parts.items():
    #             kd_parts_accumulate[k] = kd_parts_accumulate.get(k, 0.0) + float(v.item())

    #     kd_total = kd_total / self.steps
    #     total_loss += self.lambda_kd * kd_total

    #     # === 3. 记录 KD 每项 ===
    #     for k, v_sum in kd_parts_accumulate.items():
    #         parts[f"kd_{k}"] = (v_sum / self.steps) * self.lambda_kd
    #     parts["kd_stepwise"] = float(kd_total.item()) * self.lambda_kd
    #     parts["kd"] = float((self.lambda_kd * kd_total).item())

    #     return total_loss, parts

class WaveletKDDiffusion_EM_step_otf(nn.Module):
    def __init__(self, base_ckpt_path=None, base_net=None,
                 steps=1, lambda_kd=0.5, levels=2, use_edge_mask=False,
                 # ==== OTF 开关与权重 ====
                 use_onfly=True, p_aug=0.5, alpha_aug=0.5,
                 # ==== 直接透传给 add_smoke_endoscopic_onfly 的参数（键名与其一致） ====
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

        # ==== OTF 控制 ====
        self.use_onfly = use_onfly      # 训练脚本里 warmup 后自动打开
        self.p_aug = p_aug              # 触发 OTF 的概率
        self.alpha_aug = alpha_aug      # OTF 分支损失权重

        # 以 dict 形式保存，训练时原样传入 add_smoke_endoscopic_onfly
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

        # === 小波频域 gating 模块 ===
        self.freq_ch = 3 * 3 * levels  # 3通道 × (LH/HL/HH) × levels
        self.freq_gating = ResidualFrequencyGating(img_ch=3, freq_ch=self.freq_ch)

        # === 学生网络 ===
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3 + self.freq_ch, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

    # -------- 前向传播（与原 step 版一致） --------
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

    # -------- 损失计算：主分支 + step-wise KD + （可选）OTF 分支 --------
    def compute_loss(self, hazy, gt_clear, criterion):
        """
        criterion: CombinedLoss 实例（你的 build_combined_loss 返回的）
        返回: total_loss, parts(dict: 仅放 float，避免 .detach() 报错)
        """
        # 1) 主分支（真实 hazy → gt）
        pred, teacher_out, step_outputs = self.forward(hazy, return_all_steps=True)

        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # 2) step-wise KD （对每一步的输出与教师输出对齐）
        kd_step_loss = 0.0
        if self.use_edge_mask:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            for s_out in step_outputs:
                kd_step_loss = kd_step_loss + F.mse_loss(s_out * edge_mask, teacher_out.detach() * edge_mask)
        else:
            for s_out in step_outputs:
                kd_step_loss = kd_step_loss + F.mse_loss(s_out, teacher_out.detach())
        kd_step_loss = kd_step_loss / float(self.steps)

        total_loss = total_loss + self.lambda_kd * kd_step_loss
        parts["kd_stepwise"] = float(kd_step_loss.item()) * self.lambda_kd

        # 3) on-the-fly 分支（直接使用现有 add_smoke_endoscopic_onfly）
        if self.use_onfly and random.random() < self.p_aug:
            with torch.no_grad():
                hazy_otf, _ = add_smoke_endoscopic_onfly(
                    gt_clear,
                    device=gt_clear.device,
                    dtype=gt_clear.dtype,
                    **self.otf_kwargs
                )
            pred_otf = self.forward(hazy_otf)  # 不需要 teacher/steps
            otf_loss, _ = criterion(pred_otf, gt_clear, inp_for_mask=hazy_otf)

            total_loss = total_loss + self.alpha_aug * otf_loss
            parts["otf_loss"] = float(otf_loss.item()) * self.alpha_aug

        return total_loss, parts

def edge_magnitude(img):
    # 返回连续的边缘强度（非二值），便于定义可导的 edge loss
    gray = img.mean(dim=1, keepdim=True)
    lap = torch.abs(F.conv2d(gray, lap_kernel.to(img.device), padding=1))
    return lap  # [B,1,H,W]


# ---------- PTTD: Prompt Generation Module (PGM) ----------
class PromptGen(nn.Module):
    """
    轻量 PGM：从图像生成每通道的均值/方差扰动提示（delta_mu, delta_logsigma）
    这里对 3 通道图像层面做统计自适应，C=3，成本极低。
    """
    def __init__(self, in_ch=3, hidden=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden, in_ch * 2)  # 输出 [delta_mu(3), delta_logsigma(3)]

    def forward(self, x):
        feat = self.backbone(x)
        g = self.pool(feat).flatten(1)         # [B, hidden]
        out = self.head(g)                      # [B, 6]
        delta_mu, delta_logsigma = out.chunk(2, dim=1)  # 各 [B,3]
        return delta_mu, delta_logsigma


# ---------- PTTD: Feature Adaptation Module (FAM) ----------
class FeatureAdapt(nn.Module):
    """
    按通道做 per-image mean/std 自适应：
      x_hat = (x - mu) / (sigma + eps) * sigma' + mu'
    其中 mu' = mu + alpha * delta_mu
        sigma' = sigma * exp(beta * delta_logsigma)
    alpha, beta 为可学习的全局系数（极少量参数），TTT 时与 PGM 一起更新。
    """
    def __init__(self, num_ch=3, init_alpha=0.5, init_beta=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta  = nn.Parameter(torch.tensor(init_beta))
        self.eps = 1e-6
        self.num_ch = num_ch

    def forward(self, x, delta_mu, delta_logsigma):
        # x: [B,3,H,W] ；delta_*: [B,3]
        B, C, H, W = x.shape
        assert C == self.num_ch, "FAM 这里按输入图像 3 通道设计"
        # 原图像 per-image per-channel 统计
        mu    = x.mean(dim=(2,3), keepdim=True)               # [B,3,1,1]
        sigma = x.std(dim=(2,3), keepdim=True).clamp_min(self.eps)

        # 目标统计
        dmu   = delta_mu.view(B, C, 1, 1)
        dlogS = delta_logsigma.view(B, C, 1, 1)

        mu_p    = mu + self.alpha * dmu
        sigma_p = sigma * torch.exp(self.beta * dlogS)

        x_hat = (x - mu) / sigma * sigma_p + mu_p
        return x_hat
    


def gradient_smoothness(x):
    """简单的 TV-like 平滑项，抑制自适应过程中产生的伪纹理/振铃"""
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x.abs().mean() + grad_y.abs().mean()


# ===== 主体模型 =====
class KDDiffusion_EM_TTT(nn.Module):
    """
    KDDiffusion_EM + PTTD (PromptGen + FeatureAdapt)

    Train-time:
        - CombinedLoss(学生 vs GT) + λ_kd * KD(学生 vs Teacher)
    Test-time (TTT):
        - Edge Consistency（默认 Laplacian L1）
        - + 可选 Teacher Consistency（弱化）
        - + Smoothness Regularization（抑制统计扰动造成的抖动）
        - + 轻正则（限制 PGM/FAM 偏移）
        - + 动态权重（前期更重边缘，后期平衡）
    """
    def __init__(self, base_ckpt_path=None, base_net=None, steps=1,
                 lambda_kd=0.5, use_edge_mask=False, enable_teacher_consistency=True):
        super().__init__()
        assert 1 <= steps <= 10
        self.steps = steps
        self.lambda_kd = float(lambda_kd)
        self.is_kd = True
        self.use_edge_mask = bool(use_edge_mask)
        self.enable_teacher_consistency = bool(enable_teacher_consistency)

        # ---- Teacher (冻结) ----
        self.teacher = base_net if base_net is not None else TinyBaseNet()
        if base_ckpt_path is not None:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            new_state = {k.split(".", 1)[-1] if k.startswith(("module.", "net.", "model.")) else k: v
                         for k, v in state_dict.items()}
            self.teacher.load_state_dict(new_state, strict=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ---- Student ----
        self.refine_steps = nn.ModuleList([
            MiniRestormerRefineNet(in_ch=3, out_ch=3, dim=32, num_levels=1)
            for _ in range(self.steps)
        ])

        # ---- PTTD: Prompt + Feature Adapt（仅 TTT 更新）----
        self.pgm = PromptGen(in_ch=3, hidden=32)
        self.fam = FeatureAdapt(num_ch=3, init_alpha=0.5, init_beta=0.5)

    # ============ 实用辅助 ============
    def get_ttt_params(self):
        """仅返回 TTT 需要更新的参数（PGM/FAM）。"""
        return list(self.pgm.parameters()) + list(self.fam.parameters())

    def set_ttt_mode(self, enable: bool = True):
        """
        一键切换 TTT 模式：
          - 冻结全网，仅放开 PGM/FAM；或反之。
        """
        self.requires_grad_(False)
        if enable:
            for p in self.get_ttt_params():
                p.requires_grad = True

    # ============ 前向 ============
    def forward(self, hazy, return_teacher=False):
        # Teacher 输出（冻结）
        with torch.no_grad():
            teacher_out = self.teacher(hazy)

        # Prompt-based 统计自适应
        delta_mu, delta_logsigma = self.pgm(hazy)
        x = self.fam(hazy, delta_mu, delta_logsigma)

        # 学生迭代 refine
        for i in range(self.steps):
            delta = self.refine_steps[i](x)
            x = x + delta

        out = torch.clamp(x, 0, 1)
        if return_teacher:
            return out, teacher_out
        return out

    # ============ 训练阶段 ============
    def compute_loss(self, hazy, gt_clear, criterion):
        """
        训练时（有 GT）：
          total = CombinedLoss(pred, gt) + λ_kd * KD(pred, teacher_out)
        """
        pred, teacher_out = self.forward(hazy, return_teacher=True)

        # 主损失（支持 edge-aware mask 的 CombinedLoss）
        total_loss, parts = criterion(pred, gt_clear, inp_for_mask=hazy)

        # KD：学生对齐 Teacher
        if self.use_edge_mask:
            edge_mask = compute_edge_mask(gt_clear).expand_as(gt_clear)
            kd = F.mse_loss(pred * edge_mask, teacher_out.detach() * edge_mask)
        else:
            kd = F.mse_loss(pred, teacher_out.detach())

        total_loss = total_loss + self.lambda_kd * kd
        parts["kd"] = float(kd.item()) * self.lambda_kd
        return total_loss, parts

    # ============ 测试时自适应（TTT） ============
    @torch.no_grad()
    def _laplace(self, x):
        """对灰度图做拉普拉斯取绝对值，作为边缘强度。"""
        g = x.mean(dim=1, keepdim=True)
        return torch.abs(F.conv2d(g, lap_kernel.to(x.device), padding=1))

    def compute_ttt_loss(self,
                         hazy,
                         step_idx: int = 0,
                         total_steps: int = 5,
                         mode: str = "lap",           # "lap"（推荐）或 "mask"
                         adaptive_edge: bool = True,  # mask 模式是否使用自适应阈值
                         lambda_smooth: float = 0.10, # 平滑项系数
                         lambda_reg: float = 1e-3,    # PGM/FAM 正则
                         kd_scale: float = 0.5        # TTT阶段的 KD 衰减系数
                         ):
        """
        Test-Time Training（无 GT）：
          - mode="lap": 使用 Laplacian L1 做边缘一致性（更稳）
          - mode="mask": 使用二值边缘掩码 + MSE（与训练阶段一致）
        """
        # 允许反向传播：TTT 时只会对 PGM/FAM 求导
        pred, teacher_out = self.forward(hazy, return_teacher=True)

        # === 1) Edge consistency ===
        if mode == "lap":
            lap_pred = self._laplace(pred)
            lap_inp  = self._laplace(hazy)
            edge_consistency = F.l1_loss(lap_pred, lap_inp)
        elif mode == "mask":
            edge_mask = compute_edge_mask(hazy, adaptive=adaptive_edge).expand_as(hazy)
            edge_consistency = F.mse_loss(pred * edge_mask, hazy * edge_mask)
        else:
            raise ValueError(f"Unknown mode={mode}, choose from ['lap','mask'].")

        # === 2) Teacher consistency（弱化，或可关闭） ===
        if self.enable_teacher_consistency:
            kd_loss = F.mse_loss(pred, teacher_out.detach())
        else:
            kd_loss = pred.new_zeros(())  # 0 标量，避免分支判断

        # === 3) Smoothness regularization（抑制噪点/振铃） ===
        smooth_loss = gradient_smoothness(pred)

        # === 4) 轻正则：限制 PGM/FAM 的偏移幅度，避免统计发散 ===
        reg = pred.new_zeros(())
        for p in self.pgm.parameters():
            reg = reg + p.pow(2).sum()
        for p in self.fam.parameters():
            reg = reg + p.pow(2).sum()
        reg_loss = lambda_reg * reg / (pred.numel())  # 归一化到合理尺度

        # === 5) 动态权重：前期更重边缘，后期更平衡 ===
        t_den = max(1, total_steps - 1)
        w_edge = 1.0 + 0.5 * (1.0 - float(step_idx) / t_den)  # step=0 -> 1.5, step=end -> 1.0
        w_kd   = self.lambda_kd * kd_scale
        w_sm   = lambda_smooth

        total = w_edge * edge_consistency + w_kd * kd_loss + w_sm * smooth_loss + reg_loss
        return total
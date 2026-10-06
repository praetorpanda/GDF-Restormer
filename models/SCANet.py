import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair


###########################################################
#                Matrix Inverse Square Root
###########################################################
def isqrt_newton_schulz_autograd(A: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    计算矩阵 A 的逆平方根 A^(-1/2)，Newton-Schulz 迭代（单 batch）。
    A: [C, C]
    """
    dim = A.shape[0]
    normA = A.norm()
    Y = A / normA
    I = torch.eye(dim, dtype=A.dtype, device=A.device)
    Z = torch.eye(dim, dtype=A.dtype, device=A.device)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt


def isqrt_newton_schulz_autograd_batch(A: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    计算 batch 矩阵 A 的逆平方根 A^(-1/2)，A: [B, C, C]
    """
    batch_size, dim, _ = A.shape
    normA = A.view(batch_size, -1).norm(2, 1).view(batch_size, 1, 1)
    Y = A / normA
    I = torch.eye(dim, dtype=A.dtype, device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim, dtype=A.dtype, device=A.device).unsqueeze(0).expand_as(A)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)

    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt


###########################################################
#                   Channel Deconv (Whitening)
###########################################################
class ChannelDeconv(nn.Module):
    """
    对通道做 decorrelation + whitening 的模块（用于特征预处理）
    """

    def __init__(self, block: int, eps=1e-2, n_iter=5, momentum=0.1, sampling_stride=3):
        super().__init__()

        self.eps = eps
        self.n_iter = n_iter
        self.momentum = momentum
        self.block = block

        self.register_buffer("running_mean1", torch.zeros(block, 1))
        self.register_buffer("running_deconv", torch.eye(block))
        self.register_buffer("running_mean2", torch.zeros(1, 1))
        self.register_buffer("running_var", torch.ones(1, 1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        self.sampling_stride = sampling_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        if x.dim() == 3:
            raise RuntimeError("ChannelDeconv: unsupported 3D input.")

        N, C, H, W = x.size()
        B = self.block

        # 找到前 c 个通道用于 whitening
        c = int(C / B) * B
        if c == 0:
            raise RuntimeError("ChannelDeconv: block should be set smaller than channel.")

        # step1: remove mean
        if c != C:
            x1 = x[:, :c].permute(1, 0, 2, 3).contiguous().view(B, -1)
        else:
            x1 = x.permute(1, 0, 2, 3).contiguous().view(B, -1)

        if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
            x1_s = x1[:, :: self.sampling_stride ** 2]
        else:
            x1_s = x1

        mean1 = x1_s.mean(-1, keepdim=True)

        if self.num_batches_tracked == 0:
            self.running_mean1.copy_(mean1.detach())

        if self.training:
            self.running_mean1.mul_(1 - self.momentum).add_(mean1.detach() * self.momentum)
        else:
            mean1 = self.running_mean1

        x1 = x1 - mean1

        # step2: cov^(-1/2)
        if self.training:
            cov = x1_s @ x1_s.t() / x1_s.shape[1] + self.eps * torch.eye(
                B, dtype=x.dtype, device=x.device
            )
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

            if self.num_batches_tracked == 0:
                self.running_deconv.copy_(deconv.detach())

            self.running_deconv.mul_(1 - self.momentum).add_(deconv.detach() * self.momentum)
        else:
            deconv = self.running_deconv

        x1 = deconv @ x1

        # reshape back
        x1 = x1.view(c, N, H, W).contiguous().permute(1, 0, 2, 3)

        # normalize the remaining channels
        if c != C:
            x_tmp = x[:, c:].view(N, -1)
            if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
                x_s = x_tmp[:, :: self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2 = x_s.mean()
            var = x_s.var()

            if self.num_batches_tracked == 0:
                self.running_mean2.copy_(mean2.detach())
                self.running_var.copy_(var.detach())

            if self.training:
                self.running_mean2.mul_(1 - self.momentum).add_(mean2.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = (x[:, c:] - mean2) / (var + self.eps).sqrt()
            x1 = torch.cat([x1, x_tmp], dim=1)

        if self.training:
            self.num_batches_tracked.add_(1)

        if len(x_shape) == 2:
            x1 = x1.view(x_shape)
        return x1


###########################################################
#                   Delinear (FC Whitening)
###########################################################
class Delinear(nn.Module):
    """
    针对全连接层做 decorrelation/whitening 的替代实现。
    """

    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-5,
        n_iter: int = 5,
        momentum: float = 0.1,
        block: int = 512,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # whitening 相关
        if block > in_features:
            block = in_features
        else:
            if in_features % block != 0:
                block = math.gcd(block, in_features)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(self.block))
        self.register_buffer("running_deconv", torch.eye(self.block))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # whitening
        if self.training:
            X = input.view(-1, self.block)
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            self.running_mean.mul_(1 - self.momentum).add_(X_mean.detach() * self.momentum)

            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            cov = torch.addmm(self.eps, Id, 1.0 / X.shape[0], X.t(), X)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

            self.running_deconv.mul_(1 - self.momentum).add_(deconv.detach() * self.momentum)
        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = self.weight.view(-1, self.block) @ deconv
        if self.bias is None:
            b = - (w @ X_mean.unsqueeze(1)).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ X_mean.unsqueeze(1)).view(self.weight.shape[0], -1).sum(1)

        w = w.view(self.weight.shape)
        return F.linear(input, w, b)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


###########################################################
#                     FastDeconv
###########################################################
class FastDeconv(conv._ConvNd):
    """
    卷积前做 whitening 的 FastDeconv 卷积。
    结构来自原 SCANet 官方实现。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
        n_iter=5,
        momentum=0.1,
        block=64,
        sampling_stride=3,
        freeze=False,
        freeze_iter=100,
    ):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter = 0
        self.track_running_stats = True

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode="zeros",
        )

        if block > in_channels:
            block = in_channels
        else:
            if in_channels % block != 0:
                block = math.gcd(block, in_channels)

        if groups > 1:
            block = in_channels // groups

        self.block = block
        self.num_features = kernel_size[0] ** 2 * block

        if groups == 1:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_deconv", torch.eye(self.num_features))
        else:
            self.register_buffer("running_mean", torch.zeros(kernel_size[0] ** 2 * in_channels))
            self.register_buffer(
                "running_deconv", torch.eye(self.num_features).repeat(in_channels // block, 1, 1)
            )

        self.sampling_stride = sampling_stride * stride[0]
        self.freeze_iter = freeze_iter
        self.freeze = freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        B = self.block
        frozen = self.freeze and (self.counter > self.freeze_iter)

        if self.training and self.track_running_stats:
            self.counter += 1
            self.counter %= (self.freeze_iter * 10)

        if self.training and (not frozen):
            # im2col
            if self.kernel_size[0] > 1:
                X = F.unfold(
                    x,
                    self.kernel_size,
                    self.dilation,
                    self.padding,
                    self.sampling_stride,
                ).transpose(1, 2)
            else:
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[:: self.sampling_stride ** 2, :]

            if self.groups == 1:
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(
                    -1, self.num_features
                )
            else:
                X = X.view(-1, X.shape[-1])

            # subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # cov & inverse sqrt
            if self.groups == 1:
                Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                cov = torch.addmm(self.eps, Id, 1.0 / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(
                    self.num_features, dtype=X.dtype, device=X.device
                ).expand(self.groups, self.num_features, self.num_features)
                cov = torch.baddbmm(self.eps, Id, 1.0 / X.shape[1], X.transpose(1, 2), X)
                deconv = isqrt_newton_schulz_autograd_batch(cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum).add_(X_mean.detach() * self.momentum)
                self.running_deconv.mul_(1 - self.momentum).add_(deconv.detach() * self.momentum)
        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        # weight whitening
        if self.groups == 1:
            w = (
                self.weight.view(-1, self.num_features, C // B)
                .transpose(1, 2)
                .contiguous()
                .view(-1, self.num_features)
                @ deconv
            )
            b = self.bias - (w @ X_mean.unsqueeze(1)).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C // B, -1, self.num_features) @ deconv
            b = self.bias - (w @ X_mean.view(-1, self.num_features, 1)).view(self.bias.shape)

        w = w.view(self.weight.shape)
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


###########################################################
#                  Deformable Conv 2D
###########################################################
class DeformConv2d(nn.Module):
    """
    经典 Deformable Convolution（DCNv1 风格）。
    """

    def __init__(
        self,
        inc,
        outc,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=None,
        modulation=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(
            inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride
        )
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(
                inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride
            )
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # 原实现只是缩放梯度，这里保持不变
        grad_input = tuple(g * 0.1 for g in grad_input if g is not None)
        grad_output = tuple(g * 0.1 for g in grad_output if g is not None)
        return grad_input

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing="ij",
        )
        p_n = torch.cat([p_n_x.reshape(-1), p_n_y.reshape(-1)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing="ij",
        )
        p_0_x = p_0_x.reshape(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.reshape(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)

        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(1).expand(-1, c, -1, -1, -1).contiguous().view(
            b, c, -1
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [
                x_offset[..., s : s + ks].contiguous().view(b, c, h, w * ks)
                for s in range(0, N, ks)
            ],
            dim=-1,
        )
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)

        q_lt = p.floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = (
            g_lt.unsqueeze(1) * x_q_lt
            + g_rb.unsqueeze(1) * x_q_rb
            + g_lb.unsqueeze(1) * x_q_lb
            + g_rt.unsqueeze(1) * x_q_rt
        )

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out


###########################################################
#              Attention Blocks (CA + SA)
###########################################################
class ChannelAttention(nn.Module):
    def __init__(self, nc, number):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nc)
        self.prelu = nn.PReLU(nc)
        self.conv2 = nn.Conv2d(nc, nc, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(nc)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(nc, number, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(number, nc, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xx = self.prelu(self.bn1(self.conv1(x)))
        xx = self.bn2(self.conv2(xx))
        se = self.gap(xx)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return se


class SpatialAttention(nn.Module):
    def __init__(self, nc, number):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nc)
        self.prelu = nn.PReLU(nc)

        self.conv2 = nn.Conv2d(nc, number, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(number)

        self.conv3 = nn.Conv2d(number, number, 3, padding=3, dilation=3, bias=False)
        self.conv4 = nn.Conv2d(number, number, 3, padding=5, dilation=5, bias=False)
        self.conv5 = nn.Conv2d(number, number, 3, padding=7, dilation=7, bias=False)

        self.fc1 = nn.Conv2d(number * 4, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.prelu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))

        a1 = x2
        a2 = self.conv3(x2)
        a3 = self.conv4(x2)
        a4 = self.conv5(x2)

        se = torch.cat([a1, a2, a3, a4], dim=1)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))

        return se


class ResidualBlockAttn(nn.Module):
    def __init__(self, nc, number=4):
        super().__init__()
        self.CA = ChannelAttention(nc, number)
        self.SA = SpatialAttention(nc, number)

    def forward(self, x):
        x1 = self.CA(x) * x
        x2 = self.SA(x1) * x1
        return x + x2


###########################################################
#                     Residual Block
###########################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)


###########################################################
#                     Generator
###########################################################
class Generator(nn.Module):
    """
    原 SCANet Generator 完整实现：
      - FastDeconv 预处理
      - 注意力分支（AGN）
      - 主干 U-Net (down / res / DCN / up)
    """

    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        in_features=32,
        n_residual_att=6,
        n_residual_blocks=6,
    ):
        super().__init__()

        # 预处理: FastDeconv
        self.deconv = FastDeconv(input_nc, input_nc, 3, padding=1)

        # Attention Branch (AGN)
        att = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, in_features // 2, 7),
            nn.BatchNorm2d(in_features // 2),
            nn.PReLU(),
        ]
        for _ in range(n_residual_att):
            att.append(ResidualBlockAttn(in_features // 2))

        att += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features // 2, 1, 7),
            nn.Sigmoid(),
        ]
        self.att = nn.Sequential(*att)

        # Encoder Head
        conv1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, in_features, 7),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
        ]
        for _ in range(3):
            conv1 += [
                nn.Conv2d(in_features, in_features, 3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.PReLU(),
            ]
        self.conv1 = nn.Sequential(*conv1)

        # U-Net Backbone
        model = []
        c = in_features

        # Down-sample ×2
        for _ in range(2):
            model += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(c * 2),
                nn.PReLU(),
            ]
            c *= 2

        # Residual Blocks
        for _ in range(n_residual_blocks):
            model.append(ResidualBlock(c))

        # Deformable Conv ×2
        for _ in range(2):
            model += [
                DeformConv2d(c, c),
                nn.BatchNorm2d(c),
                nn.PReLU(),
            ]

        # Up-sample ×2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    c, c // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(c // 2),
                nn.PReLU(),
            ]
            c //= 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, output_nc, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

        # alpha controls attention fusion
        self.alpha = nn.Parameter(torch.tensor(0.25))
        self.gamma_max = 0.1
        self.gamma_min = 0.01

    def forward(self, x, m_gt=None, l_mask=0.0):
        """
        x: 输入 hazy 图像，范围 [0,1] / 标准化后
        m_gt: (可选) 外部提供的 GT 注意力（用于 SCL）
        l_mask: 当前 mask loss 数值，用于调度 beta
        """
        # beta: semi-curricular learning
        if l_mask >= self.gamma_max:
            beta = 1.0
        elif l_mask >= self.gamma_min:
            beta = (l_mask - self.gamma_min) / (self.gamma_max - self.gamma_min)
        else:
            beta = 0.0

        # FastDeconv 预处理
        x_dec = self.deconv(x)

        # Attention branch
        m_g = self.att(x_dec)  # [B,1,H,W]

        if m_gt is None:
            m = m_g
        else:
            m = beta * m_gt + (1.0 - beta) * m_g

        # Encoder head
        feat = self.conv1(x_dec)

        # Attention fusion
        feat_inp = self.alpha * m * feat + (1.0 - self.alpha) * feat

        # U-Net backbone
        out = self.model(feat_inp)

        # tanh 输出到 [-1,1] → 映射回 [0,1]
        out = (out + 1.0) / 2.0
        return out, m_g


###########################################################
#                   Discriminator
###########################################################
class Discriminator(nn.Module):
    """
    PatchGAN 判别器（与原 SCANet 论文一致）
    """

    def __init__(self, input_nc=3):
        super().__init__()
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)  # [B,1,H',W']
        # 论文原实现是 avgpool + sigmoid
        x = torch.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1))
        return x


###########################################################
#                        Test
###########################################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(3, 3).to(device)
    D = Discriminator(3).to(device)

    x = torch.randn(1, 3, 256, 256, device=device)
    out, m = G(x)
    d_real = D(out)

    print("Input:", x.shape)
    print("Out  :", out.shape)
    print("Mask :", m.shape)
    print("D(out):", d_real.shape)

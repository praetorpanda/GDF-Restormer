import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_

##########################################################################
## Layer Normalization
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Windowing functions (from MAXIM)
def block_images_einops(x, patch_size):  # n, h, w, c
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """Patches to images."""
    x = rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


##########################################################################
## Multi-Axis Attention (Restormer-style with block/grid partitioning)
class MultiAxisAttention(nn.Module):
    """Multi-axis attention that can work on both block and grid partitions."""
    def __init__(self, dim, num_heads, bias, block_size, grid_size, is_grid_attention=False):
        super(MultiAxisAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.block_size = block_size
        self.grid_size = grid_size
        self.is_grid_attention = is_grid_attention
        
        self.qkv = nn.Linear(dim, dim*3, bias=bias)
        self.project_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        # x shape: [B, H, W, C]
        B, H, W, C = x.shape
        
        if self.is_grid_attention:
            # Grid attention - partition into grids
            fh, fw = H // self.grid_size[0], W // self.grid_size[1]
            x = block_images_einops(x, patch_size=(fh, fw))  # [B, gh*gw, fh*fw, C]
            num_windows = self.grid_size[0] * self.grid_size[1]
            window_size = fh * fw
        else:
            # Block attention - partition into blocks
            gh, gw = H // self.block_size[0], W // self.block_size[1]
            x = block_images_einops(x, patch_size=(self.block_size[0], self.block_size[1]))  # [B, gh*gw, fh*fw, C]
            num_windows = gh * gw
            window_size = self.block_size[0] * self.block_size[1]
        
        # Reshape for attention computation
        B_new, n_windows, n_tokens, C = x.shape
        x = x.reshape(B * n_windows, n_tokens, C)  # [B*n_windows, n_tokens, C]
        
        # QKV computation
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = rearrange(out, 'b head n c -> b n (head c)')
        out = self.project_out(out)
        
        # Reshape back
        out = out.reshape(B, n_windows, window_size, C)
        
        if self.is_grid_attention:
            out = unblock_images_einops(out, grid_size=(self.grid_size[0], self.grid_size[1]), 
                                       patch_size=(fh, fw))
        else:
            out = unblock_images_einops(out, grid_size=(gh, gw), 
                                       patch_size=(self.block_size[0], self.block_size[1]))
        
        return out


##########################################################################
## Gated-Dconv Feed-Forward Network (from Restormer)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, 
                               padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
   

##########################################################################
## Multi-Axis Transformer Block (combines block and grid attention)
class MultiAxisTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, 
                 block_size, grid_size):
        super(MultiAxisTransformerBlock, self).__init__()
        
        # Block attention path
        self.norm1 = nn.LayerNorm(dim)
        self.block_attn = MultiAxisAttention(dim, num_heads, bias, block_size, grid_size, 
                                           is_grid_attention=False)
        
        # Grid attention path
        self.norm2 = nn.LayerNorm(dim)
        self.grid_attn = MultiAxisAttention(dim, num_heads, bias, block_size, grid_size, 
                                          is_grid_attention=True)
        
        # Feed-forward
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Block attention
        x = x + self.block_attn(self.norm1(x))
        
        # Grid attention
        x = x + self.grid_attn(self.norm2(x))
        
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


def window_partition_new(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size[0], win_size[0], W // win_size[1], win_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size[0], win_size[1], C) # B' ,Wh ,Ww ,C
    return windows


def window_reverse_new(windows, win_size, H, W):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size[0] / win_size[1]))
    x = windows.view(B, H // win_size[0], W // win_size[1], win_size[0], win_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        
        if norm:
            layers.append(nn.BatchNorm2d(out_channel)) #torch.nn.LayerNorm
        if relu:
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)#(B*Nw,num_heads,win_size*win_size,C/num_heads)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1] 
        k = torch.cat((k_d,k_e),dim=2)
        v = torch.cat((v_d,v_e),dim=2)
        return q,k,v
    

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww (8,8)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02) #parameter initialization
        '''
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        '''
        if token_projection =='linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    
    def update_win_size(self, win_size=None):
        if win_size is not None:
            self.win_size = win_size

    def forward(self, x, attn_kv=None, mask=None):
        # x_windows: nW*B, win_size*win_size, C
        # mask: nW*(B=1), win_size*win_size,win_size*win_size
        B_, N, C = x.shape
        #(B*Nw,num_heads,win_size*win_size,C/num_heads) i.e. (B*Nw,num_heads,token_len,head_dim)
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B*Nw,num_heads,token_len,token_len)
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative coordinates among each two  tokens
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift value range from [-7,7] to [0,14]
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        #self.register_buffer("relative_position_index", relative_position_index)
        
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww) i.e. (num_heads,token_len,token_len)
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
        
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            # (B,Nw,num_heads,token_len,token_len)    mask: (1,Nw,1,token_len,token_len)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            # (B*Nw,num_heads,token_len,token_len)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # v:(B*Nw,num_heads,token_len,head_dim)
        # x: (B*Nw,token_len,num_heads,head_dim) -> nW*B, win_size*win_size, C
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x
        
   
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x       


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        bs, hh,ww, c = x.size() # B,H,W,C
        x = x.view(bs, hh * ww, c)

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = ww)

        x = self.linear2(x)

        return x.view(bs, hh, ww, c)
    

class ScSeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=(4,16), idx=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads # 1
        self.win_size = win_size #(4,16)
        self.idx = idx # 0, win_size/2
        self.mlp_ratio = mlp_ratio # 4
        self.token_mlp = token_mlp #'leff','ffn'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=self.win_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # drop rate of DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        self.conv_fft = nn.Sequential(
            BasicConv(dim*2, dim*2, kernel_size=1, stride=1, relu=True),
            BasicConv(dim*2, dim*2, kernel_size=1, stride=1, relu=False)
        )
        self.win_pos_embed = nn.Conv3d(dim, dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=dim)
        
    def forward(self, x):
        #mask: 1,1,H,W
        B, C, H, W = x.shape # B,C,H,W
        
        ## computing shift_size
        shift_size=[0,0] if (self.idx % 2 == 0) else list(ti//2 for ti in self.win_size)       
        input_resolution = (H,W)
        if input_resolution[0] <= self.win_size[0]:
            shift_size[0] = 0
            self.win_size = (input_resolution[0], self.win_size[1])
            self.attn.update_win_size(win_size=self.win_size)
        if input_resolution[1] <= self.win_size[1]:
            shift_size[1] = 0
            self.win_size = (self.win_size[0], input_resolution[1]) 
            self.attn.update_win_size(win_size=self.win_size)
        assert 0 <= shift_size[0] < self.win_size[0], "shift_size height must in 0-win_size_h "
        assert 0 <= shift_size[1] < self.win_size[1], "shift_size width must in 0-win_size_w"

        attn_mask = None
        
        ## shift mask
        if (shift_size[0] > 0) or (shift_size[1] > 0): # (shift_size[0] > 0) and (shift_size[1] > 0):
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            if shift_size[0] > 0:
                h_slices = (slice(0, -self.win_size[0]),
                            slice(-self.win_size[0], -shift_size[0]),
                            slice(-shift_size[0], None))
            else:
                h_slices = (slice(0, None),)
            if shift_size[1] > 0:    
                w_slices = (slice(0, -self.win_size[1]),
                            slice(-self.win_size[1], -shift_size[1]),
                            slice(-shift_size[1], None))
            else:
                w_slices = (slice(0, None),)
                
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = window_partition_new(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size[0] * self.win_size[1]) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            #import ipdb;ipdb.set_trace()
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C        
        shortcut = x
        x = x.view(B, H, W, C) # B,H,W,C
        
        # cyclic shift
        if (shift_size[0] > 0) or (shift_size[1] > 0):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # B*H/win_size*W/win_size, win_size, win_size, C 
        x_windows = window_partition_new(shifted_x, self.win_size).view(B,-1,self.win_size[0],self.win_size[1],C).permute(0,4,1,2,3).contiguous()
        x_windows = x_windows + self.win_pos_embed(x_windows)  # B, C, Nw, H, W    Nw:windows number
        
        # layer norm        
        x_windows = x_windows.permute(0,2,3,4,1).view(B,-1,C).contiguous() # B Nw*H*W C 
        x_windows = self.norm1(x_windows)
        x_windows = x_windows.view(-1, self.win_size[0] * self.win_size[1], C)  # nW*B, win_size*win_size, C
       
        # W-MSA/SW-MSA
        # x_windows: nW*B, win_size*win_size, C
        # mask: nW, win_size*win_size, win_size*win_size
        attn_windows = self.attn(x_windows, mask=attn_mask)  #attn_windows, x_windows: (nW*B, win_size*win_size, C)

        # merge windows, convert back to B,H,W,C
        attn_windows = attn_windows.view(-1, self.win_size[0], self.win_size[1], C) #nW*B, win_size,win_size, C
        shifted_x = window_reverse_new(attn_windows, self.win_size, H, W)  # B H W C

        # reverse cyclic shift
        if (shift_size[0] > 0) or (shift_size[1] > 0):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        ### FFN
        y = self.sepctral_branch(x.view(B, H, W, C).permute(0,3,1,2))#B,C,H,W
        x = self.norm2(x).view(B, H, W, C) # B,H,W,C
        #x = x + self.drop_path(self.mlp(x))# B,H,W,C
        x = x + self.drop_path(self.mlp(x)) + y.permute(0,2,3,1) # B H W C
        del attn_mask
        
        return x.permute(0, 3, 1, 2)
    
    def sepctral_branch(self,x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')# 'ortho'
        y_imag = y.imag
        y_real = y.real
        y_fft = torch.cat([y_real, y_imag], dim=1)
        y = self.conv_fft(y_fft)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return y

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows
    

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads # 1
        self.win_size = win_size #8
        self.shift_size = shift_size # 0, win_size/2
        self.mlp_ratio = mlp_ratio # 4
        self.token_mlp = token_mlp #'leff','ffn'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # drop rate of DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

    def forward(self, x):
        #mask: 1,1,H,W
        B, C, H, W = x.shape # B,C,H,W
        
        input_resolution = (H,W)
        if min(input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"
        
        attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            #import ipdb;ipdb.set_trace()
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        
        
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C) # B,H,W,C
        

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # B*H/win_size*W/win_size, win_size, win_size, C  
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        
        
        # W-MSA/SW-MSA
        # x_windows: nW*B, win_size*win_size, C
        # mask: nW, win_size*win_size, win_size*win_size
        attn_windows = self.attn(x_windows, mask=attn_mask)  #attn_windows, x_windows: (nW*B, win_size*win_size, C)

        # merge windows, convert back to B,H,W,C
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C) #nW*B, win_size,win_size, C
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        
        ### FFN
        x = self.norm2(x).view(B, H, W, C) # B,H,W,C
        x = x + self.drop_path(self.mlp(x))# B,H,W,C
        del attn_mask
        
        return x.permute(0, 3, 1, 2)
    

class HierarchicalAdaptiveWindowTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super(HierarchicalAdaptiveWindowTransformer, self).__init__()

        self.shift = ScSeTransformerBlock(dim, num_heads)

        self.lewin = LeWinTransformerBlock(dim, num_heads)

        self.alpha = nn.Parameter(torch.ones(1) / 2)

    def forward(self, x):
        x = self.alpha * self.shift(x) + (1 - self.alpha) * self.lewin(x)
        return x
    

##########################################################################
## Cross-Gating Block (from MAXIM) - Fixed for proper channel handling
class CrossGatingBlock(nn.Module):
    """Cross-gating block for multi-scale feature fusion."""
    def __init__(self, x_features, num_channels, block_size, grid_size, 
                 cin_y=0, upsample_y=True, use_bias=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        
        if upsample_y and cin_y > 0:
            # For upsampling: cin_y is the input channels before upsampling
            self.ConvTranspose_y = nn.ConvTranspose2d(cin_y, num_channels, kernel_size=2, stride=2, bias=use_bias)
        elif cin_y > 0:
            self.Conv_y = nn.Conv2d(cin_y, num_channels, kernel_size=1, stride=1, bias=use_bias)
        else:
            self.Conv_y = nn.Conv2d(x_features, num_channels, kernel_size=1, stride=1, bias=use_bias)
            
        self.Conv_x = nn.Conv2d(x_features, num_channels, kernel_size=1, stride=1, bias=use_bias)
        
        self.LayerNorm_x = nn.LayerNorm(num_channels)
        self.LayerNorm_y = nn.LayerNorm(num_channels)
        
        # Ensure we have at least 1 head
        num_heads_x = max(1, num_channels // 32)
        num_heads_y = max(1, num_channels // 32)
        
        self.MultiAxisBlock_x = MultiAxisTransformerBlock(
            dim=num_channels, num_heads=num_heads_x, ffn_expansion_factor=2.66,
            bias=use_bias, LayerNorm_type='WithBias', block_size=block_size, grid_size=grid_size
        )
        
        self.MultiAxisBlock_y = MultiAxisTransformerBlock(
            dim=num_channels, num_heads=num_heads_y, ffn_expansion_factor=2.66,
            bias=use_bias, LayerNorm_type='WithBias', block_size=block_size, grid_size=grid_size
        )
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, y):
        # Process inputs
        if self.upsample_y and hasattr(self, 'ConvTranspose_y'):
            y = self.ConvTranspose_y(y)
        else:
            y = self.Conv_y(y)
        x = self.Conv_x(x)
        
        assert y.shape == x.shape, f"Shape mismatch: x {x.shape} vs y {y.shape}"
        
        shortcut_x = x
        shortcut_y = y
        
        # Cross gating with multi-axis attention
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        y = y.permute(0, 2, 3, 1)
        
        x = self.LayerNorm_x(x)
        y = self.LayerNorm_y(y)
        
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        y = y.permute(0, 3, 1, 2)
        
        # Get gating weights using multi-axis attention
        gx = torch.sigmoid(self.MultiAxisBlock_x(x))
        gy = torch.sigmoid(self.MultiAxisBlock_y(y))
        
        # Apply cross gating
        y = y * gx  # Gate y using x
        x = x * gy  # Gate x using y
        
        x = self.dropout(x)
        y = self.dropout(y)
        
        x = x + y + shortcut_x + shortcut_y
        
        return x, y


##########################################################################
class RSFMNet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=32,
        num_blocks=[4, 6, 6, 8], 
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        block_size=(8, 8),
        grid_size=(8, 8),
        dropout_rate=0.0,
    ):
        super(RSFMNet, self).__init__()

        assert len(num_blocks) == len(heads), "num_blocks and heads must have same length"
        self.depth = len(num_blocks)  # determine number of levels

        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        encoder_levels = []
        down_modules = []
        dims = [dim * (2**i) for i in range(self.depth)]

        in_dim = dim
        for i in range(self.depth):
            encoder = nn.Sequential(*[
                HierarchicalAdaptiveWindowTransformer(dim=dims[i], num_heads=heads[i])
                for _ in range(num_blocks[i])
            ])
            encoder_levels.append(encoder)

            if i < self.depth - 1:
                down = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.PixelUnshuffle(2)
                )
                down_modules.append(down)

        self.encoder_levels = nn.ModuleList(encoder_levels)
        self.down_modules = nn.ModuleList(down_modules)

        # Latent
        self.latent = nn.Sequential(*[
            MultiAxisTransformerBlock(dim=dims[-1], num_heads=heads[-1], ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias, LayerNorm_type=LayerNorm_type, block_size=block_size, grid_size=grid_size)
            for _ in range(num_blocks[-1])
        ])

        # build decoders and skip connections
        self.cross_gatings = nn.ModuleList()
        self.up_modules = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in reversed(range(1, self.depth)):
            self.cross_gatings.append(CrossGatingBlock(
                x_features=dims[i-1], num_channels=dims[i-1],
                block_size=block_size, grid_size=grid_size,
                cin_y=dims[i], upsample_y=True, use_bias=bias, dropout_rate=dropout_rate
            ))

            self.up_modules.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i]*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2)
            ))

            self.reduce_chans.append(nn.Conv2d(dims[i], dims[i-1], kernel_size=1, bias=bias))

            self.decoders.append(nn.Sequential(*[
                MultiAxisTransformerBlock(dim=dims[i-1], num_heads=heads[i-1], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type, block_size=block_size, grid_size=grid_size)
                for _ in range(num_blocks[i-1])
            ]))

        # Final refinement and output
        self.refinement = nn.Sequential(*[
            MultiAxisTransformerBlock(dim=dims[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias, LayerNorm_type=LayerNorm_type, block_size=block_size, grid_size=grid_size)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dims[0], out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        enc_feats = []
        x = self.patch_embed(inp_img)
        for i in range(self.depth):
            x = self.encoder_levels[i](x)
            enc_feats.append(x)
            if i < self.depth - 1:
                x = self.down_modules[i](x)

        x = self.latent(x)

        for i in range(self.depth - 1):
            up = self.up_modules[i](x)
            skip_enc, up = self.cross_gatings[i](enc_feats[-(i+2)], x)
            x = torch.cat([skip_enc, up], dim=1)
            x = self.reduce_chans[i](x)
            x = self.decoders[i](x)

        x = self.refinement(x)
        return self.output(x) + inp_img
  

if __name__ == "__main__":
    model = RSFMNet()
    inp = torch.randn(1, 3, 256, 256)  # Example input
    with torch.no_grad():
        # Forward pass
        # inp = inp.repeat(1, 3, 1, 1)
        out = model(inp)
        print(out.shape)  # Should be [1, 3, 64, 64]
# models/sedt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

# ---------------------------
# Frequency Domain Modulator
# ---------------------------
class FrequencyDomainModulator(nn.Module):
    def __init__(self, channels: int, win_size=8,
                 use_channel_affine=True,
                 use_band_mask=True,
                 use_spatial_gate=True,
                 use_freq_conv=True,
                 norm_type='layer'):
        super().__init__()
        self.channels = channels
        self.win_size = win_size
        self.use_channel_affine = use_channel_affine
        self.use_band_mask = use_band_mask
        self.use_spatial_gate = use_spatial_gate
        self.use_freq_conv = use_freq_conv

        if use_channel_affine:
            self.alpha_real = nn.Parameter(torch.zeros(channels))
            self.alpha_imag = nn.Parameter(torch.zeros(channels))

        if use_band_mask:
            self.thr_low  = nn.Parameter(torch.ones(channels) * 0.2)
            self.thr_high = nn.Parameter(torch.ones(channels) * 0.6)
            self.temp     = nn.Parameter(torch.ones(channels) * 8.0)
            self.alpha_low  = nn.Parameter(torch.zeros(channels))
            self.alpha_mid  = nn.Parameter(torch.zeros(channels))
            self.alpha_high = nn.Parameter(torch.zeros(channels))

        if use_freq_conv:
            self.freq_conv = nn.Conv2d(2*channels, 2*channels, 1)
            self.freq_act  = nn.GELU()

        if use_spatial_gate:
            self.spatial_conv = nn.Conv2d(channels, channels, 1)

        if norm_type == 'layer':
            self.norm = nn.LayerNorm(channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(channels)
        else:
            self.norm = None

    def forward(self, x):
        # x: [B, N, C] or [B, C, win, win]
        if x.dim() == 3:
            B, N, C = x.shape
            win = self.win_size
            x = x.transpose(1,2).contiguous().view(B, C, win, win)
        else:
            B, C, win, win_ = x.shape
            assert win == win_

        f = torch.fft.fft2(x, norm='ortho')
        real, imag = f.real, f.imag

        if self.use_channel_affine:
            ar = self.alpha_real.view(1, -1, 1, 1)
            ai = self.alpha_imag.view(1, -1, 1, 1)
            real = real * (1 + ar)
            imag = imag * (1 + ai)

        if self.use_band_mask:
            device = x.device
            wx, wy = torch.meshgrid(
                torch.linspace(-1, 1, win, device=device),
                torch.linspace(-1, 1, win, device=device),
                indexing='ij'
            )
            r = torch.sqrt(wx**2 + wy**2)[None, None, :, :]
            tl = self.thr_low.view(1, -1, 1, 1)
            th = self.thr_high.view(1, -1, 1, 1)
            tm = self.temp.view(1, -1, 1, 1)
            m_low = torch.sigmoid(tm * (tl - r))
            m_high = torch.sigmoid(tm * (r - th))
            m_mid = (1.0 - m_low) * (1.0 - m_high)
            scale = 1 \
                + self.alpha_low.view(1, -1, 1, 1)  * m_low \
                + self.alpha_mid.view(1, -1, 1, 1)  * m_mid \
                + self.alpha_high.view(1, -1, 1, 1) * m_high
            real = real * scale
            imag = imag * scale

        if self.use_freq_conv:
            freq = torch.cat([real, imag], dim=1)
            freq = self.freq_act(self.freq_conv(freq))
            real, imag = freq.chunk(2, dim=1)

        x_mod = torch.fft.ifft2(torch.complex(real, imag), norm='ortho').real

        if self.use_spatial_gate:
            gate = torch.sigmoid(self.spatial_conv(x_mod))
            x_mod = x_mod * gate

        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                x_mod = x_mod.permute(0,2,3,1)
                x_mod = self.norm(x_mod)
                x_mod = x_mod.permute(0,3,1,2)
            else:
                x_mod = self.norm(x_mod)

        x_mod = x_mod.view(B, C, win*win).transpose(1,2).contiguous()
        return x_mod

    def flops(self, win_h: int, win_w: int):
        C = self.channels
        N = win_h * win_w
        fft_ops = 2 * 5 * N * math.log2(N) * 4 * C
        affine = 2 * N * C if self.use_channel_affine else 0
        band   = 3 * N * C if self.use_band_mask else 0
        freqcv = 2 * C * N if self.use_freq_conv else 0
        spgate = 2 * C * N if self.use_spatial_gate else 0
        norm   = 2 * C * N if self.norm is not None else 0
        return int(fft_ops + affine + band + freqcv + spgate + norm)

# ---------------------------
# QKV projections
# ---------------------------
class SepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act_layer=nn.ReLU):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = kernel_size, stride
    def forward(self, x):
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        return x
    def flops(self, HW):
        return HW*self.in_channels*self.kernel_size**2/self.stride**2 + HW*self.in_channels*self.out_channels

class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)
    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n)); w = int(math.sqrt(n))
        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        q = self.to_q(x); k = self.to_k(attn_kv); v = self.to_v(attn_kv)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v
    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        return self.to_q.flops(q_L) + self.to_k.flops(kv_L) + self.to_v.flops(kv_L)

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim, self.inner_dim = dim, inner_dim
    def forward(self, x, attn_kv=None):
        B, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        kv = self.to_kv(attn_kv).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        return q, k, v
    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        return q_L*self.dim*self.inner_dim + kv_L*self.dim*self.inner_dim*2

# ---------------------------
# Attention blocks
# ---------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*win_size[0]-1)*(2*win_size[1]-1), num_heads)
        )
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2*self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        else:
            raise ValueError("Unknown token_projection")
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(self.win_size[0]*self.win_size[1], self.win_size[0]*self.win_size[1], -1) \
            .permute(2, 0, 1).contiguous()
        ratio = attn.size(-1) // rel_bias.size(-1)
        rel_bias = repeat(rel_bias, 'h l c -> h l (c d)', d=ratio)
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, H, W):
        N = self.win_size[0]*self.win_size[1]
        nW = H*W//N
        flops = self.qkv.flops(H*W, H*W)
        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N  # q@k^T
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)  # attn@v
        flops += nW * N * self.dim * self.dim                               # proj
        return flops

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, q_num, kv_num):
        flops = self.qkv.flops(q_num, kv_num)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num  # q@k^T
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num  # attn@v
        flops += q_num * self.dim * self.dim                                     # proj
        return flops

# ---------------------------
# FFN
# ---------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features, self.hidden_features, self.out_features = in_features, hidden_features, out_features
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x
    def flops(self, H, W):
        return H*W*self.in_features*self.hidden_features + H*W*self.hidden_features*self.out_features

class eca_layer_1d(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv  = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim, self.hidden_dim = dim, hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()
    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        x = self.eca(x)
        return x
    def flops(self, H, W):
        return H*W*self.dim*self.hidden_dim + H*W*self.hidden_dim*3*3 + H*W*self.hidden_dim*self.dim

# ---------------------------
# Window ops
# ---------------------------
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0,3,1,2)
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1), stride=win_size)
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size).permute(0,2,3,1).contiguous()
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, win_size, win_size, C)
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0,5,3,4,1,2).contiguous()
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1), stride=win_size)
    else:
        x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

# ---------------------------
# Down/Up, IO Proj
# ---------------------------
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.in_channel, self.out_channel = in_channel, out_channel
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L)); W = int(math.sqrt(L))
        x = x.transpose(1,2).contiguous().view(B, C, H, W)
        return self.conv(x).flatten(2).transpose(1,2).contiguous()
    def flops(self, H, W):
        return (H//2)*(W//2)*self.in_channel*self.out_channel*4*4

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.in_channel, self.out_channel = in_channel, out_channel
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L)); W = int(math.sqrt(L))
        x = x.transpose(1,2).contiguous().view(B, C, H, W)
        return self.deconv(x).flatten(2).transpose(1,2).contiguous()
    def flops(self, H, W):
        return (H*2)*(W*2)*self.in_channel*self.out_channel*2*2

class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
                                  act_layer(inplace=True))
        self.in_channel, self.out_channel = in_channel, out_channel
    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x).flatten(2).transpose(1,2).contiguous()
    def flops(self, H, W):
        return H*W*self.in_channel*self.out_channel*3*3

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2)
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L)); W = int(math.sqrt(L))
        x = x.transpose(1,2).view(B, C, H, W)
        return self.proj(x)
    def flops(self, H, W):
        # in_channel/out_channel는 생성 시점 파라미터에서 결정되므로 여기서는 상수 취급(상세 필요시 수정)
        return H*W*3*3

# ---------------------------
# LeWin Block & U-shaped backbone
# ---------------------------
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 modulator=False, cross_modulator=False, freq_modulator=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size

        self.modulator = nn.Embedding(win_size*win_size, dim) if modulator else None
        self.freq_modulator = FrequencyDomainModulator(dim, win_size) if freq_modulator else None
        self.cross_modulator = None  # 간소화: cross 모듈 제거

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            token_projection=token_projection, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise ValueError("Unknown token_mlp")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L)); W = int(math.sqrt(L))

        # mask for shifted windows (optional)
        attn_mask = None
        if self.shift_size > 0:
            shift_mask = torch.zeros((1, H, W, 1), dtype=x.dtype, device=x.device)
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
            shift_mask_windows = window_partition(shift_mask, self.win_size).view(-1, self.win_size*self.win_size)
            attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2)) if self.shift_size > 0 else x

        # window partition
        x_windows = window_partition(shifted_x, self.win_size).view(-1, self.win_size*self.win_size, C)

        # add positional modulator (optional)
        if self.modulator is not None:
            x_windows = self.with_pos_embed(x_windows, self.modulator.weight)

        # frequency modulation (optional)
        if self.freq_modulator is not None:
            x_windows = x_windows + self.freq_modulator(x_windows)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2)) if self.shift_size > 0 else shifted_x
        x = x.view(B, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        H, W = self.input_resolution
        flops = self.dim * H * W  # norm1
        if self.freq_modulator is not None:
            nW = (H*W) // (self.win_size*self.win_size)
            flops += nW * self.freq_modulator.flops(self.win_size, self.win_size)
        flops += self.attn.flops(H, W)
        flops += self.dim * H * W  # norm2
        flops += self.mlp.flops(H, W)
        return flops

class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn', shift_flag=True,
                 modulator=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    win_size=win_size, shift_size=0 if (i % 2 == 0) else win_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, token_projection=token_projection,
                    token_mlp=token_mlp, modulator=modulator
                ) for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    win_size=win_size, shift_size=0, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, token_projection=token_projection,
                    token_mlp=token_mlp, modulator=modulator
                ) for i in range(depth)
            ])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x

    def flops(self):
        return sum(blk.flops() for blk in self.blocks)

# ---------------------------
# SEDT (U-shaped Transformer)
# ---------------------------
class SEDT(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2,2,2,2,2,2,2,2,2], num_heads=[1,2,4,8,16,16,8,4,2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=True,
                 **kwargs):
        super().__init__()
        self.num_enc_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # IO
        self.input_proj  = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        self.output_proj_half    = OutputProj(in_channel=4*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        self.output_proj_quarter = OutputProj(in_channel=8*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(embed_dim, embed_dim, (img_size, img_size), depths[0], num_heads[0], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, enc_dpr[:depths[0]],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)

        self.encoderlayer_1 = BasicUformerLayer(embed_dim*2, embed_dim*2, (img_size//2, img_size//2), depths[1], num_heads[1], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, enc_dpr[depths[0]:sum(depths[:2])],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        self.encoderlayer_2 = BasicUformerLayer(embed_dim*4, embed_dim*4, (img_size//4, img_size//4), depths[2], num_heads[2], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        self.encoderlayer_3 = BasicUformerLayer(embed_dim*8, embed_dim*8, (img_size//8, img_size//8), depths[3], num_heads[3], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.conv = BasicUformerLayer(embed_dim*16, embed_dim*16, (img_size//16, img_size//16), depths[4], num_heads[4], win_size,
                                      mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, conv_dpr,
                                      norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)

        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicUformerLayer(embed_dim*16, embed_dim*16, (img_size//8, img_size//8), depths[5], num_heads[5], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dec_dpr[:depths[5]],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)

        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = BasicUformerLayer(embed_dim*8, embed_dim*8, (img_size//4, img_size//4), depths[6], num_heads[6], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dec_dpr[depths[5]:sum(depths[5:7])],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)

        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = BasicUformerLayer(embed_dim*4, embed_dim*4, (img_size//2, img_size//2), depths[7], num_heads[7], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)

        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(embed_dim*2, embed_dim*2, (img_size, img_size), depths[8], num_heads[8], win_size,
                                                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer, use_checkpoint, token_projection, token_mlp, shift_flag, modulator)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        x = x['in_img']
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask);  pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask);  pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask);  pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask);  pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4); deconv0 = torch.cat([up0, conv3], -1); deconv0 = self.decoderlayer_0(deconv0, mask=mask)
        up1 = self.upsample_1(deconv0); deconv1 = torch.cat([up1, conv2], -1); deconv1 = self.decoderlayer_1(deconv1, mask=mask)
        up2 = self.upsample_2(deconv1); deconv2 = torch.cat([up2, conv1], -1); deconv2 = self.decoderlayer_2(deconv2, mask=mask)
        up3 = self.upsample_3(deconv2); deconv3 = torch.cat([up3, conv0], -1); deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output (3 scales for deep supervision)
        y_full    = self.output_proj(deconv3)
        y_half    = self.output_proj_half(deconv2)
        y_quarter = self.output_proj_quarter(deconv1)

        x_half    = F.interpolate(x, scale_factor=0.5,  mode='bilinear', align_corners=False)
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        return [x + y_full, x_half + y_half, x_quarter + y_quarter]

    def flops(self):
        flops = 0
        R = self.reso
        flops += self.input_proj.flops(R, R)
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(R, R)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(R//2, R//2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(R//4, R//4)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(R//8, R//8)
        flops += self.conv.flops()
        flops += self.upsample_0.flops(R//16, R//16) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(R//8,  R//8)  + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(R//4,  R//4)  + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(R//2,  R//2)  + self.decoderlayer_3.flops()
        flops += self.output_proj.flops(R, R)
        return flops

            
if __name__ == "__main__":
    depths = [2,2,2,2,2,2,2,2,2]
    model = SEDT(img_size=256, embed_dim=16, depths=depths, win_size=8, token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
    print(model)
    print('# params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('GFLOPs: %.2f G' % (model.flops() / 1e9))

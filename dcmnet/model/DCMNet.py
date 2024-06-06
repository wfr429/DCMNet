import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import timm
import torchvision.ops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torch.nn import Module, Conv2d, Parameter, Softmax


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=out_channels, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=out_channels, bias=False),
            norm_layer(out_channels)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=out_channels, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.2,
                 stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBN(in_features, hidden_features, 1, stride=1, bias=True)
        self.act = act_layer()
        self.fc2 = ConvBN(hidden_features, out_features, 1, stride=1, bias=True)
        self.drop = nn.Dropout(drop, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Scan_FocusAttention(nn.Module):
    def __init__(self,
                 dim=32,
                 num_heads=8,
                 qkv_bias=True,
                 window_size=4,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.proj = ConvBN(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout()

        self.gaze = SeparableConvBN(dim, dim, kernel_size=3)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] = relative_coords[:, :, 0].clone() + self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] = relative_coords[:, :, 1].clone() + self.ws - 1
            relative_coords[:, :, 0] = relative_coords[:, :, 0].clone() * (2 * self.ws - 1)
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots = relative_position_bias.unsqueeze(0) + dots.clone()

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        v_gaze = rearrange(v, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                           d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        v_gaze = v_gaze + self.gaze(v_gaze)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out + v_gaze)
        out = self.proj(out)
        out = self.proj_drop(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class SIM(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv1 = ConvBNReLU(dim, dim // 16, 3, 2)

        self.pool_v = nn.AdaptiveAvgPool2d((1, None))  # 1xw
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # wx1

        self.conv2 = ConvBNReLU(dim // 16, dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_v = self.pool_v(x)
        x_h = self.pool_h(x)
        x = x_h @ x_v
        x = self.conv2(x)

        return x


class Block(nn.Module):
    def __init__(self, dim=64, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.4, attn_drop=0.2,
                 drop_path=0.4, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.glance_gaze_attn = Scan_FocusAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                     window_size=window_size)

        self.sim = SIM(dim)
        self.neighbor = Conv2d(dim, dim, window_size, padding=window_size // 2, stride=1)

        self.out = SeparableConvBNReLU(dim, dim, kernel_size=3)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h, w = x.size()[-2:]
        x_sim = self.sim(x)
        x_neighbor = self.neighbor(self.norm(x))
        x_neighbor = F.interpolate(x_neighbor, size=(h, w), mode='bilinear', align_corners=False)
        x = self.drop_path(x_neighbor)

        x = self.drop_path(self.glance_gaze_attn(self.norm(x)))
        x = x + self.mlp(self.norm(x))

        x = self.out(x + x_sim)

        return x


class TransEncoder(nn.Module):
    def __init__(self, dim=64):
        super(TransEncoder, self).__init__()
        self.block1 = Block(dim)
        self.conv1 = ConvBNReLU(3, dim, 3, stride=2)
        self.block2 = Block(dim)
        self.conv2 = ConvBNReLU(dim, dim, 3, stride=2)
        self.block3 = Block(dim)
        self.conv3 = ConvBNReLU(dim, dim, 3, stride=2)
        self.block4 = Block(dim)
        self.conv4 = ConvBNReLU(dim, dim, 3, stride=2)

    def forward(self, x):
        x1 = self.block1(self.conv1(x))
        x2 = self.block2(self.conv2(x1))
        x3 = self.block3(self.conv3(x2))
        x4 = self.block4(self.conv4(x3))

        return x1, x2, x3, x4


class FeatureFuse(nn.Module):
    def __init__(self, tencoder_channels=32, cencoder_channels=64):
        super(FeatureFuse, self).__init__()
        self.swt = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 SeparableConvBN(cencoder_channels, cencoder_channels // 16, kernel_size=1),
                                 nn.ReLU6(),
                                 SeparableConvBN(cencoder_channels // 16, cencoder_channels, kernel_size=1),
                                 nn.Sigmoid())

        self.conv1 = SeparableConvBN(tencoder_channels, cencoder_channels, stride=2)

        self.drop = nn.Dropout(0.4, inplace=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, xcnn, xswt):
        xswt1 = self.conv1(xswt)
        xswt2 = self.swt(xswt1) * xcnn
        x = xswt1 + xswt2 + xcnn
        x = self.relu(self.drop(x))

        return x


class Encoder(nn.Module):
    def __init__(self, cencoder_channels=(64, 128, 256, 512), tencoder_channels=32, dropout=0.):
        super(Encoder, self).__init__()
        self.transencoder = TransEncoder(tencoder_channels)

        self.fs1 = FeatureFuse(tencoder_channels, cencoder_channels[0])
        self.fs2 = FeatureFuse(tencoder_channels, cencoder_channels[1])
        self.fs3 = FeatureFuse(tencoder_channels, cencoder_channels[2])
        self.fs4 = FeatureFuse(tencoder_channels, cencoder_channels[3])

        self.init_weight()

    def forward(self, x, cx1, cx2, cx3, cx4):
        tx1, tx2, tx3, tx4 = self.transencoder(x)

        ed1 = self.fs1(cx1, tx1)
        ed2 = self.fs2(cx2, tx2)

        ed3 = self.fs3(cx3, tx3)

        ed4 = self.fs4(cx4, tx4)

        return ed1, ed2, ed3, ed4

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


class KAttention(nn.Module):
    def __init__(self, dim, scale=8, eps=1e-6):
        super(KAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_channel = dim
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = Conv(in_channels=dim, out_channels=dim // scale, kernel_size=1)
        self.key_conv = Conv(in_channels=dim, out_channels=dim // scale, kernel_size=1)
        self.value_conv = Conv(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()

class CAttention(Module):
    def __init__(self):
        super(CAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class LightAttention(nn.Module):
    def __init__(self, dim):
        super(LightAttention, self).__init__()
        self.katt = KAttention(dim)
        self.catt = CAttention()

    def forward(self, x):
        return self.katt(x) + self.catt(x)


class CorssattentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, drop=0.2):
        super(CorssattentionBlock, self).__init__()

        self.conv = SeparableConvBN(dim, dim, kernel_size=3, stride=1)
        self.crossatten = LightAttention(dim)
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)

    def forward(self, x, y):
        x_short = x
        x1 = self.norm(self.conv(x))
        ax1 = self.crossatten(x1)

        y_short = y
        y1 = self.norm(self.conv(y))
        ay1 = self.crossatten(y1)

        x2 = x1 * ay1
        y2 = y1 * ax1

        x2 = self.mlp(self.norm(x2))
        y2 = self.mlp(self.norm(y2))

        y3 = y_short + y2
        x3 = x_short + x2 + y3
        return x3, y3


class DownConnection(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, dilation=1, padding=1,
                 norm_layer=nn.BatchNorm2d):
        super(DownConnection, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding,
                      groups=out_channels, bias=False),
            norm_layer(out_channels),
            nn.Dropout(0.3, inplace=True),
            nn.ReLU(inplace=True))

class UpConnection(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(UpConnection, self).__init__()
        self.conv = SeparableConvBNReLU(in_channels, out_channels, dilation=times*2)
        self.up = nn.UpsamplingNearest2d(scale_factor=times)
        self.drop = nn.Dropout(0.3, inplace=True)
        self.gelu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        x = self.gelu(self.drop(x))

        return x

class CRWF(nn.Module):
    def __init__(self, dim=64, eps=1e-8):
        super(CRWF, self).__init__()

        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, kernel_size=3)

    def forward(self, x1, x2, x3):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * x1 + fuse_weights[1] * x2 + fuse_weights[2] * x3
        x = self.post_conv(x)
        return x



class Corssattention(nn.Module):
    def __init__(self, cencoder_channels=(64, 128, 256, 512), dropout=0.3):
        super(Corssattention, self).__init__()

        self.wf1 = CRWF(cencoder_channels[0] // 4)
        self.wf2 = CRWF(cencoder_channels[1] // 4)
        self.wf3 = CRWF(cencoder_channels[2] // 4)
        self.wf4 = CRWF(cencoder_channels[3] // 4)

        self.down12_pre = DownConnection(cencoder_channels[0], cencoder_channels[1] // 4, 2)
        self.down13_pre = DownConnection(cencoder_channels[0], cencoder_channels[2] // 4, 4)
        self.down14_pre = DownConnection(cencoder_channels[0], cencoder_channels[3] // 4, 8)

        self.up21_pre = UpConnection(cencoder_channels[1], cencoder_channels[0] // 4, 2)
        self.down23_pre = DownConnection(cencoder_channels[1], cencoder_channels[2] // 4, 2)
        self.down24_pre = DownConnection(cencoder_channels[1], cencoder_channels[3] // 4, 4)

        self.up31_pre = UpConnection(cencoder_channels[2], cencoder_channels[0] // 4, 4)
        self.up32_pre = UpConnection(cencoder_channels[2], cencoder_channels[1] // 4, 2)
        self.down34_pre = DownConnection(cencoder_channels[2], cencoder_channels[3] // 4, 2)

        self.up41_pre = UpConnection(cencoder_channels[3], cencoder_channels[0] // 4, 8)
        self.up42_pre = UpConnection(cencoder_channels[3], cencoder_channels[1] // 4, 4)
        self.up43_pre = UpConnection(cencoder_channels[3], cencoder_channels[2] // 4, 2)

        self.down12_aft = DownConnection(cencoder_channels[0] // 4, cencoder_channels[1] // 4, 2)
        self.down13_aft = DownConnection(cencoder_channels[0] // 4, cencoder_channels[2] // 4, 4)
        self.down14_aft = DownConnection(cencoder_channels[0] // 4, cencoder_channels[3] // 4, 8)

        self.up21_aft = UpConnection(cencoder_channels[1] // 4, cencoder_channels[0] // 4, 2)
        self.down23_aft = DownConnection(cencoder_channels[1] // 4, cencoder_channels[2] // 4, 2)
        self.down24_aft = DownConnection(cencoder_channels[1] // 4, cencoder_channels[3] // 4, 4)

        self.up31_aft = UpConnection(cencoder_channels[2] // 4, cencoder_channels[0] // 4, 4)
        self.up32_aft = UpConnection(cencoder_channels[2] // 4, cencoder_channels[1] // 4, 2)
        self.down34_aft = DownConnection(cencoder_channels[2] // 4, cencoder_channels[3] // 4, 2)

        self.up41_aft = UpConnection(cencoder_channels[3] // 4, cencoder_channels[0] // 4, 8)
        self.up42_aft = UpConnection(cencoder_channels[3] // 4, cencoder_channels[1] // 4, 4)
        self.up43_aft = UpConnection(cencoder_channels[3] // 4, cencoder_channels[2] // 4, 2)

        self.cross1 = CorssattentionBlock(cencoder_channels[0] // 4)
        self.cross2 = CorssattentionBlock(cencoder_channels[1] // 4)
        self.cross3 = CorssattentionBlock(cencoder_channels[2] // 4)
        self.cross4 = CorssattentionBlock(cencoder_channels[3] // 4)

        self.conv1 = SeparableConvBNReLU(cencoder_channels[0], cencoder_channels[0] // 4, kernel_size=1)
        self.conv2 = SeparableConvBNReLU(cencoder_channels[1], cencoder_channels[1] // 4, kernel_size=1)
        self.conv3 = SeparableConvBNReLU(cencoder_channels[2], cencoder_channels[2] // 4, kernel_size=1)
        self.conv4 = SeparableConvBNReLU(cencoder_channels[3], cencoder_channels[3] // 4, kernel_size=1)

        self.fuse1 = SeparableConvBNReLU(cencoder_channels[0], cencoder_channels[0], kernel_size=3)
        self.fuse2 = SeparableConvBNReLU(cencoder_channels[1], cencoder_channels[1], kernel_size=3)
        self.fuse3 = SeparableConvBNReLU(cencoder_channels[2], cencoder_channels[2], kernel_size=3)
        self.fuse4 = SeparableConvBNReLU(cencoder_channels[3], cencoder_channels[3], kernel_size=3)

        self.init_weight()

    def forward(self, ed1, ed2, ed3, ed4):
        cross1_ed2_in = self.up21_pre(ed2)
        cross1_ed3_in = self.up31_pre(ed3)
        cross1_ed4_in = self.up41_pre(ed4)
        cross1_x_in = self.conv1(ed1)
        cross1_y_in = self.wf1(cross1_ed2_in, cross1_ed3_in, cross1_ed4_in)
        cross1_x_out, cross1_y_out = self.cross1(cross1_x_in, cross1_y_in)

        cross2_ed1_in = self.down12_pre(ed1)
        cross2_ed3_in = self.up32_pre(ed3)
        cross2_ed4_in = self.up42_pre(ed4)
        cross2_x_in = self.conv2(ed2)
        cross2_y_in = self.wf2(cross2_ed1_in, cross2_ed3_in, cross2_ed4_in)
        cross2_x_out, cross2_y_out = self.cross2(cross2_x_in, cross2_y_in)

        cross3_ed1_in = self.down13_pre(ed1)
        cross3_ed2_in = self.down23_pre(ed2)
        cross3_ed4_in = self.up43_pre(ed4)
        cross3_x_in = self.conv3(ed3)
        cross3_y_in = self.wf3(cross3_ed1_in, cross3_ed2_in, cross3_ed4_in)
        cross3_x_out, cross3_y_out = self.cross3(cross3_x_in, cross3_y_in)

        cross4_ed1_in = self.down14_pre(ed1)
        cross4_ed2_in = self.down24_pre(ed2)
        cross4_ed4_in = self.down34_pre(ed3)
        cross4_x_in = self.conv4(ed4)
        cross4_y_in = self.wf4(cross4_ed1_in, cross4_ed2_in, cross4_ed4_in)
        cross4_x_out, cross4_y_out = self.cross4(cross4_x_in, cross4_y_in)

        cross2_dd1 = self.up21_aft(cross2_y_out)
        cross3_dd1 = self.up31_aft(cross3_y_out)
        cross4_dd1 = self.up41_aft(cross4_y_out)
        dd1 = self.fuse1(torch.cat((cross1_x_out, cross2_dd1, cross3_dd1, cross4_dd1), dim=1))

        cross1_dd2 = self.down12_aft(cross1_y_out)
        cross3_dd2 = self.up32_aft(cross3_y_out)
        cross4_dd2 = self.up42_aft(cross4_y_out)
        dd2 = self.fuse2(torch.cat((cross2_x_out, cross1_dd2, cross3_dd2, cross4_dd2), dim=1))

        cross1_dd3 = self.down13_aft(cross1_y_out)
        cross2_dd3 = self.down23_aft(cross2_y_out)
        cross4_dd3 = self.up43_aft(cross4_y_out)
        dd3 = self.fuse3(torch.cat((cross3_x_out, cross1_dd3, cross2_dd3, cross4_dd3), dim=1))

        cross1_dd4 = self.down14_aft(cross1_y_out)
        cross2_dd4 = self.down24_aft(cross2_y_out)
        cross3_dd4 = self.down34_aft(cross3_y_out)
        dd4 = self.fuse4(torch.cat((cross4_x_out, cross1_dd4, cross2_dd4, cross3_dd4), dim=1))

        return dd1, dd2, dd3, dd4

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ProgressiveUP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProgressiveUP, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = (self.linear(x)).permute(0, 2, 1).reshape(batch_size, -1, height, width)
        x = self.up(x)

        return x

class WF(nn.Module):
    def __init__(self, dim=128, eps=1e-8):
        super(WF, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, cencoder_channels=(64, 128, 256, 512), num_classes=8, dropout=0.3):
        super(Decoder, self).__init__()
        self.up43 = ProgressiveUP(cencoder_channels[3], cencoder_channels[2])
        self.up32 = ProgressiveUP(cencoder_channels[2], cencoder_channels[1])
        self.up21 = ProgressiveUP(cencoder_channels[1], cencoder_channels[0])

        self.wf3 = WF(cencoder_channels[2])
        self.wf2 = WF(cencoder_channels[1])
        self.wf1 = WF(cencoder_channels[0])

        self.segmentation_head = ProgressiveUP(cencoder_channels[0], num_classes)

        self.drop = nn.Dropout(dropout, inplace=True)

        self.init_weight()

    def forward(self, dd1, dd2, dd3, dd4):

        dd4_3 = self.up43(dd4)
        dd3 = self.wf3(dd3, dd4_3)
        dd3_2 = self.up32(dd3)
        dd2 = self.wf2(dd2, dd3_2)
        dd2_1 = self.up21(dd2)
        dd1 = self.wf1(dd1, dd2_1)

        out = self.drop(dd1)
        out = self.segmentation_head(out)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DCMNet(nn.Module):
    def __init__(self,
                 cencoder_channels=(64, 128, 256, 512),
                 tencoder_channels=16,
                 dropout=0.2,
                 num_classes=8,
                 backbone_name='swsl_resnet18',
                 pretrained=True
                 ):
        super(DCMNet, self).__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        self.encoder = Encoder(cencoder_channels, tencoder_channels, dropout)
        self.cross = Corssattention(cencoder_channels)
        self.decoder = Decoder(cencoder_channels, num_classes, dropout)

    def forward(self, x):
        h, w = x.size()[-2:]
        cx1, cx2, cx3, cx4 = self.backbone(x)
        ed1, ed2, ed3, ed4 = self.encoder(x, cx1, cx2, cx3, cx4)
        dd1, dd2, dd3, dd4 = self.cross(ed1, ed2, ed3, ed4)
        out = self.decoder(dd1, dd2, dd3, dd4)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out


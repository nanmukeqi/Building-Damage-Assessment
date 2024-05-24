import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models

from .senet import se_resnext50_32x4d, senet154
from .convnext import ConvNeXt
from .dpn import dpn92
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .MLFC2 import MLFC,ResPath



class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x





class SeResNext50_Unet_Loc(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_Loc, self).__init__()

        encoder_filters = [96, 192, 384, 768]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        # self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        # encoder = se_resnext50_32x4d(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        # self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        # self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        # self.conv3 = encoder.layer2
        # self.conv4 = encoder.layer3
        # self.conv5 = encoder.layer4

        self.convnext = ConvNeXt(in_chans=3, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                                 drop_path_rate=0.,
                                 layer_scale_init_value=1e-6, head_init_scale=1., pretrain=True)
        weights = torch.load("/mnt/d/daima/xview2_1st_place_solution-master/convnext_tiny_22k_224.pth", map_location="cpu")[
            "model"]
        weights.pop("head.weight")
        weights.pop("head.bias")
        self.convnext.load_state_dict(weights, strict=False)


        drop_path_rate=0.
        depths = [1, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embed4 = OverlapPatchEmbed(img_size=16, patch_size=3, stride=2, in_chans=encoder_filters[-1],
                                              embed_dim=1536)

        self.patch_embed5 = OverlapPatchEmbed(img_size=32, patch_size=3, stride=2, in_chans=encoder_filters[-1],
                                              embed_dim=1536)
        self.block2 = nn.ModuleList([shiftedBlock(
            dim=1536, num_heads=1, mlp_ratio=1, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=dpr[1], norm_layer=nn.LayerNorm,
            sr_ratio=8)])
        self.norm4 = nn.LayerNorm(1536)


        self.decoder1 = nn.Conv2d(1536, 768, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(768)
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=768, num_heads=1, mlp_ratio=1, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=dpr[0], norm_layer=nn.LayerNorm,
            sr_ratio=8)])

        # self.AFNB1 = AFNPBlock(in_channels=768, value_channels=1024, key_channels=1024)
        self.mlfc1 = MLFC(encoder_filters[-4], encoder_filters[-3], encoder_filters[-2], encoder_filters[-1],lenn=1)
        self.mlfc2 = MLFC(encoder_filters[-4], encoder_filters[-3], encoder_filters[-2], encoder_filters[-1],lenn=1)
        self.mlfc3 = MLFC(encoder_filters[-4], encoder_filters[-3], encoder_filters[-2],encoder_filters[-1], lenn=1)
        # self.rspth1 = ResPath(encoder_filters[-4], 3)
        # self.rspth2 = ResPath(encoder_filters[-3], 2)
        # self.rspth3 = ResPath(encoder_filters[-2], 1)


    def forward(self, x):
        batch_size, C, H, W = x.shape

        # enc1 = self.conv1(x)
        # enc2 = self.conv2(enc1)
        # enc3 = self.conv3(enc2)
        # enc4 = self.conv4(enc3)
        # enc5 = self.conv5(enc4)
        enc, enc_downsample = self.convnext(x)
        #
        # enc_downsample[0] = self.rspth1(enc_downsample[0])
        # enc_downsample[1] = self.rspth2(enc_downsample[1])
        # enc_downsample[2] = self.rspth3(enc_downsample[2])


        # enc_downsample[0], enc_downsample[1], enc_downsample[2] ,enc_downsample[3] = self.mlfc1(enc_downsample[0], enc_downsample[1], enc_downsample[2], enc_downsample[3] )
        # enc_downsample[0], enc_downsample[1], enc_downsample[2] ,enc_downsample[3]= self.mlfc2(enc_downsample[0], enc_downsample[1], enc_downsample[2],enc_downsample[3])
        # enc_downsample[0], enc_downsample[1], enc_downsample[2] = self.mlfc3(enc_downsample[0], enc_downsample[1], enc_downsample[2])

        # MLP block
        if H == 16:
            out, H, W = self.patch_embed4(enc)
        else:
            out, H, W = self.patch_embed5(enc)

        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(batch_size, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, enc)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = out.reshape(batch_size, H, W, -1).permute(0, 3, 1, 2).contiguous()


        dec6 = self.conv6(F.interpolate(out, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc_downsample[2]
                                       ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc_downsample[1]
                                       ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc_downsample[0]
                                       ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeResNext50_Unet_Double(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_Double, self).__init__()

        encoder_filters = [96, 192, 384, 768]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        self.convnext = ConvNeXt(in_chans=3, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                                 drop_path_rate=0.,
                                 layer_scale_init_value=1e-6, head_init_scale=1., pretrain=True)
        weights = \
        torch.load("/mnt/d/daima/xview2_1st_place_solution-master/convnext_tiny_22k_224.pth", map_location="cpu")[
            "model"]
        weights.pop("head.weight")
        weights.pop("head.bias")
        self.convnext.load_state_dict(weights, strict=False)

    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc, enc_downsample = self.convnext(x)

        dec6 = self.conv6(F.interpolate(enc, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc_downsample[2]
                                       ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc_downsample[1]
                                       ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc_downsample[0]
                                       ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

    def forward(self, x):

        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


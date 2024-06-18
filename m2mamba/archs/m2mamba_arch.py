import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ChannelAttention(nn.Module):
    def __init__(self, base_filters, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        kernel_size = 1
        conv = default_conv
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(base_filters, base_filters // squeeze_factor, kernel_size),
            nn.ReLU(inplace=True),
            conv(base_filters // squeeze_factor, base_filters, kernel_size),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return y * x


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        kernel_size = 3
        conv = default_conv
        if is_light_sr:  # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            conv(num_feat, num_feat // compress_ratio, kernel_size),
            nn.GELU(),
            conv(num_feat // compress_ratio, num_feat, kernel_size),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class CABG(nn.Module):
    # CAB group
    def __init__(self, num_feat, blk_num, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CABG, self).__init__()
        self.main = nn.Sequential(
            *[CAB(num_feat, is_light_sr, compress_ratio, squeeze_factor) for _ in range(blk_num)]
        )

    def forward(self, x):
        return self.main(x) + x


class MSMM(nn.Module):
    # Multi-scale Mamba Module
    def __init__(self, hidden_dim, small_size, large_size, input_size):
        super(MSMM, self).__init__()
        conv = default_conv
        self.small_conv = conv(hidden_dim, hidden_dim, kernel_size=small_size, stride=1)
        self.large_conv = conv(hidden_dim, hidden_dim, kernel_size=large_size-1, stride=large_size)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mamba1 = Mamba(
            d_model=hidden_dim,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.mamba2 = Mamba(
            d_model=hidden_dim,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.skip_scale1 = nn.Parameter(torch.ones(hidden_dim, 1, 1))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim, 1, 1))
        self.ln_3 = nn.LayerNorm(hidden_dim)

        self.liner1 = nn.Linear(int(input_size ** 2 / large_size**2), int(input_size ** 2))
        self.liner2 = nn.Linear(int(input_size ** 2 / large_size**2), int(input_size ** 2))

        self.attention = CAB(hidden_dim)

    def forward(self, input):
        b, c, w, h = input.shape

        x1 = self.ln_1(self.small_conv(input).reshape(b, -1, c))
        x1 = self.mamba1(x1)
        x2 = self.ln_2(self.large_conv(input).reshape(b, -1, c))
        x2 = self.mamba2(x2)
        x = x1 * self.liner1(x2.reshape(b, c, -1)).reshape(b, -1, c) + self.liner2(x2.reshape(b, c, -1)).reshape(b, -1, c)
        x = input * self.skip_scale1 + self.attention(self.ln_3(x).reshape(b, c, w, h))
        return x


class MSMG(nn.Module):
    # Feature Refinement Module (Mamba+CNN)
    def __init__(self, num_feat, small_size, large_size, MSMM_num=2, input_size=512):
        super(MSMG, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv

        self.main = nn.Sequential(
            *[MSMM(num_feat, small_size, large_size, input_size) for _ in range(MSMM_num)]
        )

        self.conv = nn.Sequential(
            conv(num_feat, num_feat // 2, kernel_size), act,
            conv(num_feat // 2, num_feat, kernel_size)
        )

    def forward(self, x):
        x_ = self.main(x)
        out = self.conv(x_) + x
        return out


class MSLFM(nn.Module):
    # Multi-scale Learned Fusion Module
    def __init__(self,
                 in_feature,
                 out_feature,
                 ):
        super(MSLFM, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        self.downsample = conv(in_feature, out_feature, kernel_size, stride=2)

        self.preprocess = nn.Sequential(
            conv(3, out_feature // 2, kernel_size), act,
            conv(out_feature // 2, out_feature, kernel_size)
        )

        self.postprocess = nn.Sequential(
            conv(out_feature * 2, out_feature, kernel_size), act,
            conv(out_feature, out_feature, kernel_size)
        )

        self.ca = ChannelAttention(out_feature)

    def forward(self, enc, x):
        down_enc = self.downsample(enc)

        x = self.preprocess(x)
        x = self.postprocess(torch.cat([down_enc, x], dim=1))
        att = self.ca(x)
        out = att + down_enc

        return out


class MSAFM(nn.Module):
    # Multi-scale Attention Fusion Module
    def __init__(self,
                 in_feature,
                 out_feature,
                 base_features,
                 ):
        super(MSAFM, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv

        self.upsample = nn.Sequential(
            conv(in_feature, in_feature * 4, 1, bias=False),
            nn.PixelShuffle(2),
            conv(in_feature, out_feature, kernel_size)
        )

        self.refine = nn.ModuleList([conv(base_feature, out_feature, kernel_size) for base_feature in base_features])

        self.fusion = nn.ModuleList(
            nn.Sequential(
                conv(out_feature * 2, out_feature, kernel_size), act,
                conv(out_feature, out_feature // 2, kernel_size), act,
                conv(out_feature // 2, out_feature, kernel_size)
            ) for _ in range(len(base_features))
        )

        self.main = nn.Sequential(
            conv(out_feature * 4, out_feature, kernel_size), act,
            conv(out_feature, out_feature // 2, kernel_size), act,
            conv(out_feature // 2, out_feature, kernel_size)
        )

        self.cab = CAB(out_feature)

    def forward(self, x1, x2, x3, x):
        up_x = self.upsample(x)
        x1 = self.refine[0](x1)
        x2 = self.refine[1](x2)
        x3 = self.refine[2](x3)

        fusion1 = self.fusion[0](torch.cat([x1, up_x], dim=1))
        fusion2 = self.fusion[1](torch.cat([x2, up_x], dim=1))
        fusion3 = self.fusion[2](torch.cat([x3, up_x], dim=1))

        att = self.cab(self.main(torch.cat([fusion1, fusion2, fusion3, up_x], dim=1)))
        out = att + up_x

        return out


class toImg(nn.Module):
    # feature to image
    def __init__(self,
                 in_feature,
                 out_feature=3,
                 ):
        super(toImg, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        self.recon = nn.Sequential(
            conv(in_feature, in_feature, kernel_size), act,
            conv(in_feature, in_feature, kernel_size),
        )
        self.post = conv(in_feature, out_feature, kernel_size)

    def forward(self, res, x):
        res = self.recon(res) + res
        out = self.post(res)
        return out + x


@ARCH_REGISTRY.register()
class MSMambaNet(nn.Module):
    # Multi-Scale Mamba Network
    def __init__(self,
                 in_feature=3,
                 out_feature=3,
                 input_size=512,
                 CAB_num=3,
                 MSMM_num=[2,2,2,2],
                 small_size=[3,3,3,3],
                 large_size=[32,16,8,8],
                 n_filters=32,
                 en_blocks=[1, 1, 2, 2],
                 mid_blocks=4,
                 de_blocks=[1, 1, 2, 2],
                 return_rgb=False,
                 ):
        super(MSMambaNet, self).__init__()
        # prepare
        kernel_size = 3
        conv = default_conv
        self.save_multi_imgs = return_rgb

        # [encoder] Mutil-Scale Input Fusion
        self.head = conv(in_feature, n_filters, kernel_size)

        # stage 1  (h, w)  n_filters
        self.encoder_1 = nn.Sequential(*[CABG(n_filters, CAB_num) for _ in range(en_blocks[0])])
        self.en_fusion_1 = MSLFM(n_filters, n_filters * 2)

        # stage 2 (h//2, w//2) n_filters*2
        self.encoder_2 = nn.Sequential(*[MSMG(n_filters * 2, small_size[0], large_size[0], MSMM_num[0], input_size//2) for _ in range(en_blocks[1])])
        self.en_fusion_2 = MSLFM(n_filters * 2, n_filters * 4)

        # stage 3 (h//4, w//4) n_filters*4
        self.encoder_3 = nn.Sequential(*[MSMG(n_filters * 4, small_size[1], large_size[1], MSMM_num[1], input_size//4) for _ in range(en_blocks[2])])
        self.en_fusion_3 = MSLFM(n_filters * 4, n_filters * 8)

        # stage 4 (h//8, w//8) n_filters*8
        self.encoder_4 = nn.Sequential(*[MSMG(n_filters * 8, small_size[2], large_size[2], MSMM_num[2], input_size//8) for _ in range(en_blocks[3])])

        # [middle]  Feature Refine n_filters*8
        self.middle = nn.Sequential(*[MSMG(n_filters * 8, small_size[3], large_size[3], MSMM_num[3], input_size//8) for _ in range(mid_blocks)])

        # [decoder] Multi-Scale Attention Fusion + ToImg
        base_features = [n_filters, n_filters * 2, n_filters * 4]
        # stage 4 (h//8, w//8) n_filters*8
        self.decoder_4 = nn.Sequential(*[MSMG(n_filters * 8, small_size[2], large_size[2], MSMM_num[2], input_size//8) for _ in range(de_blocks[3])])
        self.out_4 = toImg(n_filters * 8, out_feature)

        # stage 3 (h//4, w//4) n_filters*4
        self.de_fusion_3 = MSAFM(n_filters * 8, n_filters * 4, base_features)
        self.decoder_3 = nn.Sequential(*[MSMG(n_filters * 4, small_size[1], large_size[1], MSMM_num[1], input_size//4) for _ in range(de_blocks[2])])
        self.out_3 = toImg(n_filters * 4, out_feature)

        # stage 2 (h//2, w//2) n_filters*2
        self.de_fusion_2 = MSAFM(n_filters * 4, n_filters * 2, base_features)
        self.decoder_2 = nn.Sequential(*[MSMG(n_filters * 2, small_size[0], large_size[0], MSMM_num[0], input_size//2) for _ in range(de_blocks[1])])
        self.out_2 = toImg(n_filters * 2, out_feature)

        # stage 1 (h, w) n_filters
        self.de_fusion_1 = MSAFM(n_filters * 2, n_filters, base_features)
        self.decoder_1 = nn.Sequential(*[CABG(n_filters, CAB_num) for _ in range(de_blocks[0])])
        self.out_1 = toImg(n_filters, out_feature)

    def forward(self, x, **kwargs):
        x_1 = x
        x_2 = F.interpolate(x_1, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_8 = F.interpolate(x_4, scale_factor=0.5)

        outputs = list()

        x = self.head(x)
        res_en_1 = self.encoder_1(x) + x
        x = self.en_fusion_1(res_en_1, x_2)

        res_en_2 = self.encoder_2(x) + x
        x = self.en_fusion_2(res_en_2, x_4)

        res_en_3 = self.encoder_3(x) + x
        x = self.en_fusion_3(res_en_3, x_8)

        res_en_4 = self.encoder_4(x) + x

        res_en_1_2 = F.interpolate(res_en_1, scale_factor=0.5)
        res_en_1_3 = F.interpolate(res_en_1_2, scale_factor=0.5)
        res_en_2_1 = F.interpolate(res_en_2, scale_factor=2)
        res_en_2_3 = F.interpolate(res_en_2, scale_factor=0.5)
        res_en_3_2 = F.interpolate(res_en_3, scale_factor=2)
        res_en_3_1 = F.interpolate(res_en_3_2, scale_factor=2)

        x = self.middle(res_en_4)

        res_de_4 = self.decoder_4(x) + x

        x = self.de_fusion_3(res_en_1_3, res_en_2_3, res_en_3, res_de_4)
        res_de_3 = self.decoder_3(x) + x

        x = self.de_fusion_2(res_en_1_2, res_en_2, res_en_3_2, res_de_3)
        res_de_2 = self.decoder_2(x) + x

        x = self.de_fusion_1(res_en_1, res_en_2_1, res_en_3_1, res_de_2)
        res_de_1 = self.decoder_1(x) + x

        if self.save_multi_imgs:
            outputs.append(self.out_4(res_de_4, x_8))
            outputs.append(self.out_3(res_de_3, x_4))
            outputs.append(self.out_2(res_de_2, x_2))

        outputs.append(self.out_1(res_de_1, x_1))

        # return outputs[-1], outputs

        return outputs[-1]


if __name__ == '__main__':
    device = "cuda:0"
    model = MSMambaNet().to(device)
    ipt = torch.randn(1, 3, 512, 512).to(device)
    outs = model(ipt)
    print(outs.shape)

# -*- coding: utf-8 -*-
import os
import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.functional as F
import math
from PIL import Image
from compressai.models.utils import conv


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


Backward_tensorGrid = [{} for i in range(8)]
Backward_tensorGrid_cpu = {}


def torch_warp(tensorInput, tensorFlow):
    if tensorInput.device == torch.device('cpu'):
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cpu()

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    else:
        device_id = tensorInput.device.index
        if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cuda().to(device_id)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)


def read_image(filepath):
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def cal_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                   torch_warp(im2_list[img_index], flow_up),
                                                   flow_up], 1))

        return flow


class ResBottleneckBlock(nn.Module):
    def __init__(self, channel, slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if slope < 0.0001:
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return x + out


class ResBlock1(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y)
        return x * y[:, :, None, None]


class ConvBlockResidual(nn.Module):
    def __init__(self, ch_in, ch_out, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            SELayer(ch_out) if se_layer else nn.Identity(),
        )
        self.up_dim = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up_dim(x)
        return x2 + x1


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            ResBlock1(128, 0),
            ResBlock1(128, 0),
            ResBlock1(128, 0),
            ResBlock1(128, 0),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)
        # print(x.shape, x3.shape)
        # exit()

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class RefineNet(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=64, out_ch=2):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, 3, stride=1, padding=1),
            ResBottleneckBlock(hidden_channel),
            ResBottleneckBlock(hidden_channel),
            ResBottleneckBlock(hidden_channel),
            nn.Conv2d(hidden_channel, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x, ref_frame):
        return x + self.refine(torch.cat([x, ref_frame], 1))


class Reconstruction(nn.Module):
    def __init__(self, in_ch=64, channel=64, out_ch=3, return_fea=True):
        super().__init__()
        self.return_fea = return_fea
        self.first_conv = nn.Conv2d(in_ch, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel, channel)
        self.unet_2 = UNet(channel, channel)
        self.recon_conv1 = nn.Conv2d(channel, out_ch, 3, stride=1, padding=1)
        self.recon_conv2 = nn.Conv2d(channel, out_ch, 3, stride=1, padding=1)
        self.recon_conv3 = nn.Conv2d(channel * 2, out_ch, 3, stride=1, padding=1)

        self.weight1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            ResBlock1(channel),
            nn.Conv2d(channel, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.weight2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            ResBlock1(channel),
            nn.Conv2d(channel, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feature = self.first_conv(x)
        feature1 = self.unet_1(feature)
        feature2 = self.unet_2(feature)
        recon1 = self.recon_conv1(feature1)
        recon2 = self.recon_conv2(feature2)
        recon3 = self.recon_conv3(torch.cat([feature1, feature2], 1))

        w1 = self.weight1(feature1)
        w2 = self.weight2(feature2)
        recon = w1 * recon1 + w2 * recon2 + (1 - w1 - w2) * recon3
        # print(feature.shape, recon.shape)
        if self.return_fea:
            return feature, recon
        else:
            return recon


class FeatureExtraction(nn.Module):
    def __init__(self, in_ch=6, nf=64, k=3, s=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, k, s, k // 2, bias=True)
        self.rsb1 = nn.Sequential(
            ResBlock1(nf, 0),
            ResBlock1(nf, 0),
            ResBlock1(nf, 0),
        )

    def forward(self, x):
        x = self.conv1(x)
        res1 = x + self.rsb1(x)
        return res1


class InterLayerPrediction(nn.Module):
    def __init__(self, in_ch=3, hidden=64, up_out=32, out_ch=3, fea_in=64, return_s3=False):
        super().__init__()
        self.return_s3 = return_s3
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            ResBlock1(hidden),
        )
        self.in_conv1 = nn.Sequential(
            nn.Conv2d(fea_in, hidden, 3, padding=1),
            ResBlock1(hidden),
        )

        self.d2s = nn.Sequential(
            nn.PixelShuffle(4),
            conv(16, 64, 3, 1),
        )

        self.fea_convert = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            ResBlock1(hidden),
        )

        self.fea_embd = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1),
            ResBlock1(hidden),
            ResBlock1(hidden, start_from_relu=False),
        )

        self.fea_ext = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            ResBlock1(hidden),
            ResBlock1(hidden),
            nn.Conv2d(hidden, 32, 3, padding=1),
        )

        self.out_conv = nn.Conv2d(up_out, out_ch, 3, stride=1, padding=1)

        self.weight = nn.Sequential(
            nn.Conv2d(up_out, hidden, 3, stride=1, padding=1),
            ResBlock1(hidden),
            nn.Conv2d(hidden, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.lrelu = nn.LeakyReLU(True)

    def forward(self, ref_frame, warped, mv, curr_fea, feature=None):
        if feature is None:
            fea = self.lrelu(self.in_conv(ref_frame))
        else:
            fea = self.lrelu(self.in_conv1(feature))
        fea = torch_warp(fea, mv)
        curr_fea = self.fea_convert(self.d2s(curr_fea))
        # print(fea.shape, self.d2s(curr_fea).shape)
        fea3 = self.fea_embd(torch.cat([fea, curr_fea], 1))
        up_out = self.fea_ext(fea3)

        w = self.weight(up_out)
        out = w * warped + (1 - w) * self.out_conv(up_out)

        return up_out, out


if __name__ == "__main__":
    _x1 = torch.rand((1, 3, 256, 256))
    _x2 = torch.rand((1, 2, 256, 256))
    _x3 = torch.rand((1, 64, 128, 128))
    _x4 = torch.rand((1, 256, 64, 64))
    model = MyMCNet2_1()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    c, cc = model(_x3, _x1, _x3, _x2, _x4)
    # print(c.shape, cc.shape)

    # model = ContextNet()
    # print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # c = model(_x1, _x2)
    #
    # model1 = Unet()
    # print(f'Total Parameters = {sum(p.numel() for p in model1.parameters() if p.requires_grad)}')
    # out = model1(_x1, _x2, c)
    # print(out.shape)

    # model = Warp_net()
    # print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

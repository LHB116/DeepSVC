# -*- coding: utf-8 -*-
import os
import math
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import subpel_conv3x3, conv3x3
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
import torch.nn as nn
import torch


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


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


class ChannelSplitICIP2020ResB(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3):
        super().__init__()
        self.N = N
        self.num_slices = 8
        self.max_support_slices = 4

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(in_ch, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, out_ch, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
        )

        self.h_scale_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i + 1, self.max_support_slices + 1), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, N // 2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N // 2, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        updated = self.entropy_bottleneck.update(force=force)
        if scale_table is None:
            scale_table = get_scale_table()
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class ICIP2020ResB(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


if __name__ == "__main__":
    h, w = 256, 256
    x = torch.rand((1, 3, h, w))
    x1 = torch.rand((1, 96, h // 16, w // 16))
    # h, w = 240, 416
    model = ChannelSplitICIP2020ResB(3, 96, 3).cuda()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    out = model(x.cuda(), x1.cuda())
    print(out['x_hat'].shape)


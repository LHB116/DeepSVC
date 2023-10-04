# -*- coding: utf-8 -*-
import os
from modules import ME_Spynet, RefineNet, Reconstruction, FeatureExtraction, InterLayerPrediction, torch_warp
from image_model import ChannelSplitICIP2020ResB
from compressai.entropy_models import EntropyBottleneck
import torch.nn as nn
import math
import torch
import numpy as np
from pytorch_msssim import ms_ssim


class DeepSVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.opticFlow = ME_Spynet()
        self.mv_codec = ChannelSplitICIP2020ResB(8, 64, 2)
        self.res_codec = ChannelSplitICIP2020ResB(64 + 6, 96, 64)
        self.MC = InterLayerPrediction()

        self.RefineMvNet = RefineNet(5, 64, 2)
        self.RefineResiNet = RefineNet(64 + 3, 64, 64)

        self.FeatureExtractor = FeatureExtraction(3, 64)
        self.enhance = Reconstruction(32 + 64, 64, 3, return_fea=True)

    def forward(self, ref_frame, curr_frame, sm_fea, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_loss = torch.mean((warped_frame - curr_frame).pow(2))
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)
        mc_loss = torch.mean((predict_frame - curr_frame).pow(2))

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1))
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        # print(predict_frame.shape, recon_res.shape)
        # exit()

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        # feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        # print(feature.shape, recon_image.shape)
        # exit()
        # distortion
        mse_loss = torch.mean((recon_image - curr_frame).pow(2))
        bpp = bpp_mv + bpp_res

        return recon_image, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp

    def forward1(self, ref_frame, curr_frame, sm_fea, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_loss = torch.mean((warped_frame - curr_frame).pow(2))
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)
        mc_loss = torch.mean((predict_frame - curr_frame).pow(2))

        return predict_frame, warp_loss, mc_loss, bpp_mv

    def forward_msssim(self, ref_frame, curr_frame, sm_fea, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_msssim = ms_ssim(warped_frame, curr_frame, 1.0)
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)
        mc_msssim = ms_ssim(predict_frame, curr_frame, 1.0)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1))
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        # print(predict_frame.shape, recon_res.shape)
        # exit()

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        msssim = ms_ssim(recon_image, curr_frame, 1.0)
        bpp = bpp_mv + bpp_res

        return recon_image, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp

    def compress(self, ref_frame, curr_frame, sm_fea, feature=None):

        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_out_enc = self.mv_codec.compress(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"])['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea

        res_out_enc = self.res_codec.compress(torch.cat([ref_frame, res, predict_frame], 1))
        return mv_out_enc, res_out_enc

    def decompress(self, ref_frame, mv_out_enc, res_out_enc, sm_fea, feature):

        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"])['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        recon_res = self.res_codec.decompress(res_out_enc["strings"], res_out_enc["shape"])['x_hat']
        recon_res = self.RefineResiNet(recon_res, ref_frame)

        recon_image_fea = predict_frame_fea + recon_res
        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))

        return feature, recon_image.clamp(0., 1.), warped_frame.clamp(0., 1.), predict_frame.clamp(0., 1.)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def mv_aux_loss(self):
        return sum(m.loss() for m in self.mv_codec.modules() if isinstance(m, EntropyBottleneck))

    def res_aux_loss(self):
        return sum(m.loss() for m in self.res_codec.modules() if isinstance(m, EntropyBottleneck))

    def update(self, force=False):
        updated = self.mv_codec.update(force=force)
        updated |= self.res_codec.update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        mv_codec_dict = {k[len('mv_codec.'):]: v for k, v in state_dict.items() if 'mv_codec' in k}
        res_codec_dict = {k[len('res_codec.'):]: v for k, v in state_dict.items() if 'res_codec' in k}

        self.mv_codec.load_state_dict(mv_codec_dict)
        self.res_codec.load_state_dict(res_codec_dict)

        super().load_state_dict(state_dict)


if __name__ == "__main__":
    model = DeepSVC()
    print(f'[*] Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

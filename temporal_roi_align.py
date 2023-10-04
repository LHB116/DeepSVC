# Copyright (c) OpenMMLab. All rights reserved.
# mmtracking/mmtrack/models/vid

import warnings

import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from addict import Dict

from ..builder import MODELS
from .base import BaseVideoDetector

from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import subpel_conv3x3, conv3x3
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder


import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models import build_detector
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule, Sequential

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def cal_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


# for temporal_roi_align
@MODELS.register_module()
class SELSA(BaseVideoDetector):
    def __init__(self,
                 detector,
                 pretrains=None,
                 init_cfg=None,
                 compress_ch=72,
                 fea=56,
                 beta=64,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SELSA, self).__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            detector_pretrain = pretrains.get('detector', None)
            if detector_pretrain:
                detector.init_cfg = dict(
                    type='Pretrained', checkpoint=detector_pretrain)
            else:
                detector.init_cfg = None
        self.detector = build_detector(detector)
        # --------------------  load weights  ------------------
        # download from
        # https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth
        ckpt = './weights/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth'
        tgt_model_dict = self.detector.state_dict()
        src_pretrained_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        for k, v in src_pretrained_dict.items():
            if k[len('detector.'):] not in tgt_model_dict:
                print(k)
        _pretrained_dict = {k[len('detector.'):]: v for k, v in src_pretrained_dict.items() if
                            k[len('detector.'):] in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.detector.load_state_dict(tgt_model_dict)

        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.lmbda = fea
        self.beta = beta
        mark = 0.0
        if self.lmbda == 4:  # 4
            mark = 0.0067
        elif self.lmbda == 8:
            mark = 0.013
        elif self.lmbda == 12:
            mark = 0.025
        elif self.lmbda == 16:
            mark = 0.0483

        self.backbone_teacher = ResNetTeacher()
        self.detector.backbone_p = OursResNetStudentP(N=compress_ch, cond_enty=True, refine=False, cond_ep=True)

        print(f"[*************************************] Load Pretrained Weights")
        # troialign_official_backbone.pth' is extract from  selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth
        ckpt = './weights/troialign_official_backbone.pth'
        tgt_model_dict = self.detector.backbone.state_dict()
        src_pretrained_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.detector.backbone.load_state_dict(tgt_model_dict)

        # troialign_official_backbone.pth' is extract from  selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth
        ckpt = './weights/troialign_official_backbone.pth'
        tgt_model_dict = self.detector.backbone_p.state_dict()
        src_pretrained_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        for k, v in src_pretrained_dict.items():
            if k not in tgt_model_dict:
                print(k)
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.detector.backbone_p.load_state_dict(tgt_model_dict)

        # troialign_official_backbone.pth' is extract from  selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth
        ckpt = './weights/troialign_official_backbone.pth'
        tgt_model_dict = self.backbone_teacher.state_dict()
        src_pretrained_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.backbone_teacher.load_state_dict(tgt_model_dict)

        self.i_frame_codec = ICIP2020ResB()
        # 0.0250   0.0483   0.0932   0.18
        i_frame_model_path = './weights/ICIP2020ResB/mse'  # path to your i frame model weights
        state_dict = torch.load(f'{i_frame_model_path}/lambda_{mark}.pth', map_location='cpu')
        self.i_frame_codec.load_state_dict(state_dict['state_dict'])

        self.freeze_module('i_frame_codec')
        self.i_frame_codec.eval()
        self.freeze_module('backbone_teacher')
        self.backbone_teacher.eval()
        for p in self.detector.backbone.parameters():
            p.requires_grad = False
        self.detector.backbone.eval()

        self.process = Process()
        self.mse = nn.MSELoss()
        self.my_iters = 0

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            ref_img (Tensor): of shape (N, 2, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                2 denotes there is two reference images for each input image.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): The list only has one Tensor. The
                Tensor contains ground truth bboxes for each reference image
                with shape (num_all_ref_gts, 5) in
                [ref_img_id, tl_x, tl_y, br_x, br_y] format. The ref_img_id
                start from 0, and denotes the id of reference image for each
                key image.

            ref_gt_labels (list[Tensor]): The list only has one Tensor. The
                Tensor contains class indices corresponding to each reference
                box with shape (num_all_ref_gts, 2) in
                [ref_img_id, class_indice].

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bbox.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals (None | Tensor) : override rpn proposals with custom
                proposals. Use when `with_rpn` is False.

            ref_gt_instance_ids (None | list[Tensor]): specify the instance id
                for each ground truth bboxes of reference images.

            ref_gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes of reference images can be ignored when computing the
                loss.

            ref_gt_masks (None | Tensor) : True segmentation masks for each
                box of reference image used if the architecture supports a
                segmentation task.

            ref_proposals (None | Tensor) : override rpn proposals with custom
                proposals of reference images. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(img) == 1, \
            'selsa video detector only supports 1 batch size per gpu for now.'

        N, _, H, W = img.size()
        num_pixels = N * H * W
        if self.my_iters % 20 == 0 and torch.distributed.get_rank() == 0:
            print(f'[ ** ] MY-iters: {self.my_iters}')

        if self.lmbda == 1:  # 4
            if self.my_iters > 100000:
                exit()
        elif self.lmbda == 2:
            if self.my_iters > 100000:
                exit()
        elif self.lmbda == 4:
            if self.my_iters > 80000:
                exit()
        elif self.lmbda == 8:
            if self.my_iters > 70000:
                exit()

        # if (self.my_iters + 1) % 1e3 == 0:
        #     self.detector.backbone_p.layer1.update(force=True)
        #     if torch.distributed.get_rank() == 0:
        #         print(f'[ ** ] self.detector.backbone_p.layer1.update(force=True)')

        # if self.my_iters > 5e4:
        #     self.lmbda = 0.0

        _, num_ref, _, _, _ = ref_img.size()
        # print(f'{torch.distributed.get_rank()} size:', ref_img.size())

        with torch.no_grad():
            # print(1, torch.max(ref_img[:, 0]), torch.min(ref_img[:, 0]))
            temp = self.process(ref_img[:, 0], inverse=True) / 255.0
            # print(2, torch.max(temp), torch.min(temp))
            # temp = temp / 255.0
            # temp = temp.clamp(0.0, 1.0)
            # print(3, torch.max(temp), torch.min(temp))

            out_enc = self.i_frame_codec(temp.clamp(0.0, 1.0))
            # dec_frame = out_enc['x_hat'].clamp(0.0, 1.0)
            # print(4, torch.max(dec_frame), torch.min(dec_frame))
            mse = torch.mean((temp - out_enc['x_hat'].clamp(0.0, 1.0)) ** 2)
            ipsnr = 10 * torch.log10(1.0 / mse)
            ibpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_enc["likelihoods"].values()
            )
            ref_frame = self.process(out_enc['x_hat'].clamp(0.0, 1.0) * 255.0)
            # print(5, torch.max(ref_frame), torch.min(ref_frame))
            # exit()

            # teacher_outputs_i = self.backbone_teacher(ref_frame)
            student_outputs_i = self.detector.backbone(ref_frame)  # i frame
            # student_outputs_i1 = self.detector.backbone(ref_img[:, 0])
            if self.lmbda != 0.0:
                teacher_outputs_p1 = self.backbone_teacher(img)
                if num_ref > 1:
                    teacher_outputs_p2 = self.backbone_teacher(ref_img[:, 1])

        supp = torch.cat([student_outputs_i[0], student_outputs_i[0]], 0) if self.multi_supp else student_outputs_i[0]

        student_outputs_p1 = self.detector.backbone_p(img, supp, mean=self.multi_supp)

        supp = torch.cat([student_outputs_i[0], student_outputs_p1[0]], 0) if self.multi_supp else student_outputs_p1[0]

        bpp_loss = self.detector.backbone_p.bpp_loss
        aux_loss = self.detector.backbone_p.layer1.aux_loss()
        if num_ref > 1:
            student_outputs_p2 = self.detector.backbone_p(ref_img[:, 1], supp, mean=self.multi_supp)
            bpp_loss += self.detector.backbone_p.bpp_loss
            aux_loss += self.detector.backbone_p.layer1.aux_loss()

        neck_in = torch.cat([student_outputs_p1[-1], student_outputs_i[-1], student_outputs_p2[-1]], 0) \
            if num_ref > 1 else torch.cat([student_outputs_p1[-1], student_outputs_i[-1]], 0)

        all_x = self.detector.neck([neck_in])

        # if num_ref > 1:
        #     mse_loss = self.mse(teacher_outputs_p1[-1], student_outputs_p1[-1]) + \
        #                self.mse(teacher_outputs_p2[-1], student_outputs_p2[-1]) + \
        #                self.mse(teacher_outputs_i[-1], student_outputs_i[-1])
        # else:
        #     mse_loss = self.mse(teacher_outputs_p1[-1], student_outputs_p1[-1]) + \
        #                self.mse(teacher_outputs_i[-1], student_outputs_i[-1])

        # if num_ref > 1:
        #     mse_loss = self.mse(teacher_outputs_p1[-1], student_outputs_p1[-1]) + \
        #                self.mse(teacher_outputs_p2[-1], student_outputs_p2[-1])
        # else:
        #     mse_loss = self.mse(teacher_outputs_p1[-1], student_outputs_p1[-1])
        #
        # # td_loss = mse_loss.div(num_ref) * self.lmbda + bpp_loss.div(num_ref)

        x = []
        ref_x = []
        for i in range(len(all_x)):
            x.append(all_x[i][[0]])
            ref_x.append(all_x[i][1:])

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas[0])
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes,
            gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        # print(aux_loss, mse_loss, bpp_loss, td_loss)
        # print(roi_losses)
        # {'loss_cls': tensor(0.6037, device='cuda:3', grad_fn=<MulBackward0>), 'acc': tensor([83.5821], device='cuda:3'), 'loss_bbox': tensor(0.1628, device='cuda:3', grad_fn=<MulBackward0>)}
        # print(losses.keys())
        # 'aux', 'fea_mse_loss', 'bpp_loss', 'ipsnr', 'ibpp', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox'
        # exit()

        if self.lmbda != 0.0:
            losses['aux'] = aux_loss.div(num_ref)
            losses['fea_mse'] = self.mse(teacher_outputs_p1[-1], student_outputs_p1[-1]) + \
                                self.mse(teacher_outputs_p2[-1], student_outputs_p2[-1]) if num_ref > 1 else \
                self.mse(teacher_outputs_p1[-1], student_outputs_p1[-1])
            losses['td_loss'] = losses['fea_mse'].div(num_ref) * self.lmbda + bpp_loss.div(num_ref)
            losses['bpp'] = bpp_loss.div(num_ref)
            losses['ipsnr'] = ipsnr
            losses['ibpp'] = ibpp
        else:
            losses['aux'] = aux_loss.div(num_ref)
            losses['bpp_loss'] = bpp_loss.div(num_ref)

        losses['ipsnr'] = ipsnr
        losses['ibpp'] = ibpp
        losses['loss_rpn_cls'] = losses['loss_rpn_cls'] * self.beta
        losses['loss_rpn_bbox'] = losses['loss_rpn_bbox'] * self.beta
        losses['loss_cls'] = losses['loss_cls'] * self.beta
        losses['loss_bbox'] = losses['loss_bbox'] * self.beta

        # del aux_loss
        # del bpp_loss
        # del all_x
        # del temp
        # del ref_frame
        # del teacher_outputs_p1
        # del student_outputs_p1
        # del neck_in
        # del supp
        # del self.detector.backbone_p.bpp_loss
        # gc.collect()
        # torch.cuda.empty_cache()
        self.my_iters += 1

        return losses

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        # print(img_metas)
        # exit()
        frame_id = img_metas[0].get('frame_id', -1)
        supp_fea_num = 2
        assert frame_id >= 0
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)
        ibpp, ipsnr, pfeabpp = 0.00, float('nan'), 0.00
        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas[0]
                ref_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])

            x = self.detector.extract_feat(img)
            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i] = torch.cat((ref_x[i], x[i]), dim=0)
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas.extend(img_metas)
        # test with fixed stride
        else:
            if frame_id == 0:
                self.supp = None
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas[0]

                num_pixels = ref_img[0].size(2) * ref_img[0].size(3)
                ref_num = ref_img[0].size(0)

                temp = self.process(ref_img[0][0].unsqueeze(0), inverse=True)
                temp = temp / 255.0
                temp = temp.clamp(0.0, 1.0)

                out_enc = self.i_frame_codec.compress(temp)
                out_dec = self.i_frame_codec.decompress(out_enc["strings"], out_enc["shape"])
                ibpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                # self.ref_frame1 = out_dec["x_hat"]
                rec = self.process(out_dec["x_hat"] * 255.0).repeat(ref_num, 1, 1, 1)
                ipsnr = cal_psnr(temp, out_dec["x_hat"])

                # recs, bits = [], []
                # for data in ref_img[0]:
                #     temp = self.process(data.unsqueeze(0), inverse=True)
                #     temp = temp / 255.0
                #     temp = temp.clamp(0.0, 1.0)
                #
                #     out_enc = self.i_frame_codec.compress(temp)
                #     out_dec = self.i_frame_codec.decompress(out_enc["strings"], out_enc["shape"])
                #     bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                #     bits.append(bpp)
                #     # self.ref_frame1 = out_dec["x_hat"]
                #     recs.append(self.process(out_dec["x_hat"] * 255.0))
                #     ipsnr = cal_psnr(temp, out_dec["x_hat"])
                #
                # ibpp = np.average(bits)
                # self.ref_frame = recs[0]
                # rec = torch.cat(recs, 0)

                out = self.detector.backbone(rec)
                # print(frame_id, 'ibpp', ibpp, rec.shape)
                # exit()
                # self.supp = out[0][0].unsqueeze(0)
                self.supp = out[0][0].unsqueeze(0).repeat((supp_fea_num, 1, 1, 1))
                # print(1, self.supp.shape)
                ref_x = self.detector.neck([out[-1]])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
                    x.append(ref_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x = []
                if frame_id % 12 == 0:
                    # print('i_frame', ref_img[0].shape)
                    num_pixels = ref_img[0].size(0) * ref_img[0].size(2) * ref_img[0].size(3)
                    temp = self.process(ref_img[0], inverse=True)
                    temp = temp / 255.0
                    temp = temp.clamp(0.0, 1.0)
                    out_enc = self.i_frame_codec.compress(temp)
                    out_dec = self.i_frame_codec.decompress(out_enc["strings"], out_enc["shape"])
                    ibpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                    # mse = torch.mean((ref_img[0] - out_dec["x_hat"]).pow(2)).item()
                    # psnr = 10 * np.log10(1.0 / mse)
                    # print(frame_id, 'ibpp', ibpp, psnr, mse)
                    # ref_x = self.detector.extract_feat(out_dec["x_hat"])
                    # self.ref_frame1 = out_dec["x_hat"]
                    out = self.detector.backbone(self.process(out_dec["x_hat"] * 255.0))
                    ref_x = self.detector.neck([out[-1]])
                    # self.ref_frame = self.process(out_dec["x_hat"] * 255.0)
                    # self.supp = out[0]
                    self.supp = torch.cat([self.supp, out[0]], 0)[1:]
                    # print(2, self.supp.shape)
                    # self.feature = None
                    ipsnr = cal_psnr(temp, out_dec["x_hat"])
                else:
                    # x, ref_frame, supp, encode=False
                    # print(ref_img[0].shape, self.ref_frame.shape, )

                    out = self.detector.backbone_p(x=ref_img[0], supp=self.supp, encode=True, mean=self.multi_supp)
                    ref_x = self.detector.neck([out[-1]])
                    pfeabpp = self.detector.backbone_p.bpp_loss
                    # print(frame_id, 'pbpp', pbpp)
                    # self.ref_frame = ref_img[0]

                    # temp = self.process(ref_img[0], inverse=True)
                    # temp = temp / 255.0
                    # temp = temp.clamp(0.0, 1.0)
                    # sm_fea, mv_out_enc, res_out_enc = self.p_frame_codec.compress(self.ref_frame1, temp, self.feature)
                    # self.feature, dec_p_frame, warped_frame, predict_frame = \
                    #     self.p_frame_codec.decompress(self.ref_frame1, sm_fea, mv_out_enc, res_out_enc, self.feature)
                    # ppsnr = cal_psnr(temp, dec_p_frame)
                    # presbpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                    # pmvbpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                    # self.ref_frame = self.process(dec_p_frame * 255.0)

                    # self.supp = out[0]
                    self.supp = torch.cat([self.supp, out[0]], 0)[1:]
                    # print(3, self.supp.shape)

                for i in range(len(ref_x)):
                    self.memo.feats[i] = torch.cat(
                        (self.memo.feats[i], ref_x[i]), dim=0)[1:]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img_metas.extend(ref_img_metas[0])
                self.memo.img_metas = self.memo.img_metas[1:]
            else:
                assert ref_img is None
                x = self.detector.extract_feat(img)

            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i][num_left_ref_imgs] = x[i]
                # print(11, x[i].shape)
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas[num_left_ref_imgs] = img_metas[0]

        return x, img_metas, ref_x, ref_img_metas, ibpp, pfeabpp, ipsnr

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (list[Tensor] | None): The list only contains one Tensor
                of shape (1, N, C, H, W) encoding input reference images.
                Typically these should be mean centered and std scaled. N
                denotes the number for reference images. There may be no
                reference images in some cases.

            ref_img_metas (list[list[list[dict]]] | None): The first and
                second list only has one element. The third list contains
                image information dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain 'filename',
                'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on
                the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

            proposals (None | Tensor): Override rpn proposals with custom
                proposals. Use when `with_rpn` is False. Defaults to None.

            rescale (bool): If False, then returned bboxes and masks will fit
                the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The detection results.
        """

        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            ref_img_metas = ref_img_metas[0]

        # frame_id = img_metas[0].get('frame_id', -1)
        # print('-----------------', frame_id)
        x, img_metas, ref_x, ref_img_metas, ibpp, pfeabpp, ipsnr = \
            self.extract_feats(img, img_metas, ref_img, ref_img_metas)

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas)
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        outs = self.detector.roi_head.simple_test(
            x,
            ref_x,
            proposal_list,
            ref_proposals_list,
            img_metas,
            rescale=rescale)

        results = dict()
        results['det_bboxes'] = outs[0]
        if len(outs) == 2:
            results['det_masks'] = outs[1]

        codec_result = dict()
        codec_result['ibpp'] = ibpp
        codec_result['pfeabpp'] = pfeabpp
        codec_result['ipsnr'] = ipsnr
        return results, codec_result

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# 23282688
class ResNetTeacher(BaseModule):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 1),
                 dilations=(1, 1, 1, 2),
                 out_indices=(0, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg={'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}):
        super(ResNetTeacher, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        # print(stage_blocks)
        # exit()
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
                len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)  # bn1
            self.relu = nn.ReLU(inplace=True)
            # print(self.norm1_name)
            # exit()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            # print(layer_name)
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        # for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
        #     x = layer(x)
        #     if i in self.out_indices:
        #         outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNetTeacher, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class cFeatureCompress(nn.Module):
    def __init__(self, in_ch=3, N=24, out_ch=256):
        super().__init__()
        self.N = N
        self.num_slices = 6
        self.max_support_slices = 3
        if N == 64 or N == 72:
            self.num_slices = 8
            self.max_support_slices = 4

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(in_ch + 64, 128),
            nn.LeakyReLU(True),
            ResBottleneckBlock(128),
            conv(128, 128),
            nn.LeakyReLU(True),
            ResBottleneckBlock(128),
            conv(128, 96),
            nn.LeakyReLU(True),
            ResBottleneckBlock(96),
            conv(96, N),
        )

        self.g_s = nn.Sequential(
            subpel_conv3x3(N, 96, 2),
            nn.LeakyReLU(True),
            ResBottleneckBlock(96),
            subpel_conv3x3(96, 96, 2),
            nn.LeakyReLU(True),
            ResBottleneckBlock(96),
            nn.Conv2d(96, 96, 3, 1, 1),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(True),
            conv3x3(N, N),
            nn.LeakyReLU(True),
            conv3x3(N, N, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(True),
            conv3x3(N, N),
            nn.LeakyReLU(True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(True),
            conv3x3(N, N),
        )

        self.h_scale_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(True),
            conv3x3(N, N),
            nn.LeakyReLU(True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(True),
            conv3x3(N, N),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices) + 64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices) + 64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i + 1, self.max_support_slices + 1) + 64, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, N, stride=1, kernel_size=3),
                nn.LeakyReLU(True),
                conv(N, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.fea_convert = nn.Sequential(
            conv(256, 128),
            nn.LeakyReLU(True),
            conv(128, 64),
        )

        self.fea_convert1 = nn.Sequential(
            conv(256, 128, 3, 1),
            nn.LeakyReLU(True),
            conv(128, 64, 3, 1),
        )

        self.d2s = nn.Sequential(
            nn.PixelShuffle(4),
            conv(16, 64, 3, 1),
        )

        self.g_s1 = nn.Sequential(
            conv(96 + 64, 256, 3, 1),
            nn.LeakyReLU(True),
            ResBottleneckBlock(256),
            ResBottleneckBlock(256),
        )

        # self.g_s1 = nn.Sequential(
        #     conv(96 + 64, 96 + 64, 3, 1),
        #     nn.LeakyReLU(True),
        #     ResBottleneckBlock(96 + 64),
        #     ResBottleneckBlock(96 + 64),
        #     nn.Conv2d(96 + 64, out_ch, 3, 1, 1),
        # )

    def forward(self, x, fea=None):
        y = self.g_a(torch.cat([x, self.d2s(fea)], 1))  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]
        supp_sm = self.fea_convert(fea)
        # print(latent_scales.shape, latent_means.shape, supp_sm.shape)
        # exit()

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means, supp_sm] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales, supp_sm] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat1 = self.g_s(y_hat)

        # print(x_hat1.shape, fea_in.shape, self.fea_convert1(fea).shape)
        # exit()
        x_hat = self.g_s1(torch.cat([x_hat1, self.fea_convert1(fea)], 1))

        return {
            "x_hat": x_hat, "y": y,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, fea):
        y = self.g_a(torch.cat([x, self.d2s(fea)], 1))
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        supp_sm = self.fea_convert(fea)

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

            mean_support = torch.cat([latent_means, supp_sm] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales, supp_sm] + support_slices, dim=1)
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

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "y":y}

    def decompress(self, strings, shape, fea):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        supp_sm = self.fea_convert(fea)

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
            mean_support = torch.cat([latent_means, supp_sm] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales, supp_sm] + support_slices, dim=1)
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
        x_hat1 = self.g_s(y_hat)

        # print(x_hat1.shape, fea_in.shape, self.fea_convert1(fea).shape)
        # exit()
        x_hat = self.g_s1(torch.cat([x_hat1, self.fea_convert1(fea)], 1))

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
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
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


class OursResNetStudentP(BaseModule):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 1),
                 dilations=(1, 1, 1, 2),
                 out_indices=(0, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg={'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'},
                 N=24, cond_enty=False, refine=True, cond_ep=False,
                 ):
        super(OursResNetStudentP, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        # print(stage_blocks)
        # exit()
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            if i >= 0:
                layer_name = f'layer{i + 1}'
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
                len(self.stage_blocks) - 1)

        self.layer1 = cFeatureCompress(N=N)

        self.w_refine = refine
        if self.w_refine:
            self.refine = RefineNet1()
        self.act = nn.ReLU(inplace=True)
        self.cond_enty = cond_enty

        self.bpp_loss = None

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            pass
        #     self.conv1 = build_conv_layer(
        #         self.conv_cfg,
        #         in_channels,
        #         stem_channels,
        #         kernel_size=7,
        #         stride=2,
        #         padding=3,
        #         bias=False)
        #     self.norm1_name, norm1 = build_norm_layer(
        #         self.norm_cfg, stem_channels, postfix=1)
        #     self.add_module(self.norm1_name, norm1)
        #     self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        pass

    def forward(self, x, supp=None, mean=False, encode=False):
        outs = []

        N, _, H, W = x.size()
        num_pixels = N * H * W

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            if i == 0:
                if encode:
                    out_enc = self.layer1.compress(x) if not self.cond_enty else self.layer1.compress(x, supp[-N:])
                    out_dec = self.layer1.decompress(out_enc["strings"], out_enc["shape"]) if not self.cond_enty else self.layer1.decompress(out_enc["strings"], out_enc["shape"], supp[-N:])
                    self.bpp_loss = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                    x = out_dec['x_hat']
                    # y = out_enc['y']
                    if self.w_refine:
                        x = self.refine(x, supp, mean)
                    x = self.act(x)
                else:
                    x = layer(x, fea=supp[-1].unsqueeze(0)) if self.cond_enty else layer(x)
                    self.bpp_loss = sum(
                        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                        for likelihoods in x["likelihoods"].values()
                    )
                    # y = x["y"]
                    x = x['x_hat']
                    # x, supp, ref_frame
                    if self.w_refine:
                        x = self.refine(x, supp, mean=mean)
                    x = self.act(x)
            else:
                x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
        # return outs, y

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(OursResNetStudentP, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval


class RefineNet1(nn.Module):
    def __init__(self):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
        )

    def forward(self, x, supp, mean=True):
        x1 = self.refine(x)
        x1 = x1 / x1.norm(p=2, dim=1, keepdim=True)
        supp = self.refine(supp)
        supp = supp / supp.norm(p=2, dim=1, keepdim=True)

        if mean:
            ada_weights = torch.mean(x1 * supp, dim=1, keepdim=True)
        else:
            ada_weights = torch.sum(x1 * supp, dim=1, keepdim=True)
        ada_weights = ada_weights.softmax(dim=0)
        agg_x = torch.sum(x * ada_weights, dim=0, keepdim=True)
        return x + agg_x
        # return x + self.refine(torch.cat([x, supp], 1))


class Process(torch.nn.Module):
    def __init__(self):
        super(Process, self).__init__()

    def forward(self, tenInput, inverse=False):
        # img_norm_cfg = dict(
        #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        if not inverse:
            tenBlue = (tenInput[:, 0:1, :, :] - 123.675) / 58.395
            tenGreen = (tenInput[:, 1:2, :, :] - 116.28) / 57.12
            tenRed = (tenInput[:, 2:3, :, :] - 103.53) / 57.375
        else:
            tenBlue = tenInput[:, 0:1, :, :] * 58.395 + 123.675
            tenGreen = tenInput[:, 1:2, :, :] * 57.12 + 116.28
            tenRed = tenInput[:, 2:3, :, :] * 57.375 + 103.53
        return torch.cat([tenRed, tenGreen, tenBlue], 1)

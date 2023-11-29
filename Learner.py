import os
import torch
import datetime
import utils
from utils import AverageMeter, load_pretrained
from tqdm import tqdm
from tensorboardX import SummaryWriter
from video_model import DeepSVC
from image_model import ICIP2020ResB
from semantic_layer import OursResNetStudentP, ResNetTeacher
import json
import logging
from dataset import get_dataset, get_vid_dataset
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
import numpy as np


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


class HZHTrainer_1(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 2frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        # self.stage1_step = 20  # 2frames
        # self.stage2_step = self.stage1_step + 20  # 3frames
        # self.stage3_step = self.stage2_step + 20  # 4frames
        # self.stage4_step = self.stage3_step + 20  # 5frames
        # self.stage5_step = self.stage4_step + 20  # 5frames

        self.grad_clip = 1.0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        # DVC2_MC_322WOSM_L
        # self.log_dir = os.path.join(args.log_root, f"322WOSM_L_PSNR{args.l_PSNR}_{date}")
        self.log_dir = os.path.join(args.log_root, f"322WOSM_abaltion4_PSNR{args.l_PSNR}_{date}")
        # os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        training_set, valid_set = get_dataset(args, mf=7)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # model
        self.mode_type = args.mode_type
        self.graph = DeepSVC().cuda()
        # self.graph.p_fea_model.eval()
        # for p in self.graph.p_fea_model.parameters():
        #     p.requires_grad = False
        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')

            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_psnr, train_mc_psnr = AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp = AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            # adjust learning_rate
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                # print(self.global_step)
                if self.stage3_step < self.global_step and (self.global_step) % 5e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                f = self.get_f()
                feature = None

                if 0 <= self.global_step < self.stage2_step:
                    ref_frame = frames[0]
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, feature=feature)
                        ref_frame = decoded_frame.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            warp_weight = 0.1
                            mc_weight = 0.15
                        else:
                            warp_weight = 0.0
                            mc_weight = 0.0
                        distortion = mse_loss + warp_weight * warp_loss + mc_loss * mc_weight
                        loss = self.l_PSNR * distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "ALL{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                elif self.stage2_step <= self.global_step < self.stage4_step:
                    ref_frame = frames[0]
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, feature=feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        loss = self.l_PSNR * mse_loss + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "ALL{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    ref_frame = frames[0]
                    _mse, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, feature=feature)
                        # ref_frame = decoded_frame.detach().clone()
                        _mse += mse_loss * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())
                        _loss = self.l_PSNR * mse_loss + bpp

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        # self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                        # self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                        # self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                        # self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                        # self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                        # self.writer.add_scalar('train_psnr', psrn, self.global_step)
                        # self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                        # self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                        # self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                        # self.writer.add_scalar('train_loss', _loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "Final{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                    num = f * (f - 1) // 2
                    loss = self.l_PSNR * _mse.div(num) + _bpp.div(num)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    aux_loss = _aux_loss.div(num)
                    aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 6:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.2f}|{:.2f}|{:.2f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_warp_psnr.avg,
                                    train_mc_psnr.avg,
                                    train_psnr.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            # if epoch % self.args.val_freq == 0:
            self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()
        val_warp_psnr, val_mc_psnr = AverageMeter(), AverageMeter()
        val_res_bpp, val_mv_bpp = AverageMeter(), AverageMeter()
        val_psnr, val_msssim = AverageMeter(), AverageMeter()
        val_res_aux, val_mv_aux, val_aux = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = frames[0]
                f = self.get_f()
                feature = None
                for frame_index in range(1, f):
                    curr_frame = frames[frame_index]
                    decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                        self.graph(ref_frame, curr_frame, feature)
                    feature = feature1.detach().clone()
                    ref_frame = decoded_frame.detach().clone()
                    loss = self.l_PSNR * mse_loss + bpp

                    msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                    psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                    mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                    warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                    mv_aux = self.graph.mv_aux_loss()
                    res_aux = self.graph.res_aux_loss()
                    aux = self.graph.aux_loss()

                    val_loss.update(loss.mean().detach().item(), self.batch_size)
                    val_warp_psnr.update(warp_psrn, self.batch_size)
                    val_mc_psnr.update(mc_psrn, self.batch_size)
                    val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                    val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                    val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                    val_msssim.update(msssim, self.batch_size)
                    val_psnr.update(psrn, self.batch_size)

                    val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                    val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                    val_aux.update(aux.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar('val_mv_aux', mv_aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_res_aux', res_aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_aux', aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_psnr', psrn, self.global_step)
                    self.writer.add_scalar('val_loss', loss.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_warp_psnr', warp_psrn, self.global_eval_step)
                    self.writer.add_scalar('val_mc_psnr', mc_psrn, self.global_eval_step)
                    self.writer.add_scalar('val_bpp', bpp.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_res_bpp', bpp_res.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_mv_bpp', bpp_mv.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_msssim', msssim, self.global_eval_step)
                    self.writer.add_scalar('val_psnr', psrn, self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                     "AUX[{:.2f}|{:.2f}|{:.2f}]".format(
                        self.global_epoch + 1,
                        self.num_epochs,
                        loss.mean().detach().item(),
                        warp_psrn,
                        mc_psrn,
                        psrn,
                        bpp_mv.mean().detach().item(),
                        bpp_res.mean().detach().item(),
                        bpp.mean().detach().item(),
                        mv_aux.detach().item(),
                        res_aux.detach().item(),
                        aux.detach().item(),
                    )

                self.global_eval_step += 1

                # if k > 20:
                #     break

        self.logger.info("VALID [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                         "AUX[{:.2f}|{:.2f}|{:.2f}]". \
                         format(self.global_epoch + 1,
                                self.num_epochs,
                                val_loss.avg,
                                val_warp_psnr.avg,
                                val_mc_psnr.avg,
                                val_psnr.avg,
                                val_mv_bpp.avg,
                                val_res_bpp.avg,
                                val_bpp.avg,
                                val_mv_aux.avg,
                                val_res_aux.avg,
                                val_aux.avg
                                ))
        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 4
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 7
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        last_step = checkpoint["global_step"]
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")
        self.logger.info(f"[*] Load Pretrained Model From Step {last_step}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        # print(self.global_step)
        # exit()
        del checkpoint

    def adjust_lr(self):
        # if self.global_step > self.stage2_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        if self.global_step > self.stage3_step:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        if self.global_step > self.stage4_step:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0

        if self.global_step > self.stage4_step + 6e4:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = list(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = list(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class Trainer_ICIP2020ResB_WSM_PSNR(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 3frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        self.grad_clip = 1.0
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM
        lmbda, fea_mse, beta = 0.0, 0.0, 0.0
        if self.l_PSNR == 1280:
            lmbda, fea_mse, beta = 0.0932, 64, 48
        elif self.l_PSNR == 640:
            # lmbda, fea_mse, beta = 0.0483, 32, 24
            lmbda, fea_mse, beta = 0.0483, 20, 16
        elif self.l_PSNR == 320:
            lmbda, fea_mse, beta = 0.025, 16, 12
        elif self.l_PSNR == 160:
            lmbda, fea_mse, beta = 0.013, 8, 6
        elif self.l_PSNR == 80:
            lmbda, fea_mse, beta = 0.0067, 4, 3
        else:
            print('I-lambda error')
            exit()

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"15resb_WSM_I{lmbda}_PSNR{args.l_PSNR}_{beta}{fea_mse}_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size

        training_set, valid_set = get_dataset(args, mf=6, return_orgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = int(self.stage5_step)
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # model
        self.mode_type = args.mode_type
        self.graph = DeepSVC().cuda()

        # self.logger.info(f"[*] Load Pretrained Model...")
        # ckpt = './checkpoint/LHB_DVC_WOSM_bpg2048.pth'
        # tgt_model_dict = self.graph.state_dict()
        # src_pretrained_dict = torch.load(ckpt)['state_dict']
        # _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        # tgt_model_dict.update(_pretrained_dict)
        # self.graph.load_state_dict(tgt_model_dict)

        self.i_codec = ICIP2020ResB()  # 3, 4, 5, 6
        ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/mse/lambda_{lmbda}.pth'
        self.logger.info(f'[*] Load i_codec from {ckpt}')
        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = load_pretrained(checkpoint["state_dict"])
        self.i_codec.load_state_dict(state_dict)
        self.i_codec.update(force=True)
        self.i_codec.eval()
        self.i_codec.cuda()
        for p in self.i_codec.parameters():
            p.requires_grad = False

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_i.pth'
        self.logger.info(f'[*] Load sm_i from {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_i = ResNetTeacher()
        self.sm_i.load_state_dict(pcheckpoint["state_dict"])
        self.sm_i.eval()
        self.sm_i.cuda()
        for p in self.sm_i.parameters():
            p.requires_grad = False

        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_p.pth'
        self.logger.info(f'[*] Load sm_p from {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_p = OursResNetStudentP(N=72)
        self.sm_p.load_state_dict(pcheckpoint["state_dict"])
        self.sm_p.layer1.update(force=True)
        self.sm_p.eval()
        self.sm_p.cuda()
        for p in self.sm_p.parameters():
            p.requires_grad = False

        self.process = Process()
        self.supp = None

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_psnr, train_mc_psnr = AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp, train_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            # adjust learning_rate
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                self.adjust_lr()
                if self.stage3_step < self.global_step and self.global_step % 5e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                with torch.no_grad():
                    ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                    i_mse = torch.mean((frames[0] - ref_frame).pow(2))
                    i_psnr = 10 * np.log10(1.0 / i_mse.detach().cpu())

                    smi = self.sm_i(self.process(frames[0].mul(255)))
                    self.supp = torch.cat([smi[0], smi[0]], 0)

                f = self.get_f()
                feature = None
                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        with torch.no_grad():
                            sm = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]

                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, sm[0], feature=feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            warp_weight = 0.1
                        else:
                            warp_weight = 0
                        distortion = mse_loss + warp_weight * (warp_loss + mc_loss)
                        loss = self.l_PSNR * distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                        train_i_psnr.update(i_psnr, self.batch_size)
                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T-ALL{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   i_psnr,
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _mse, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    # loss = torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        with torch.no_grad():
                            sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, y, feature=feature)
                        # ref_frame = decoded_frame.detach().clone()
                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        # tmp = torch.ones_like(i_psnr) + (i_psnr - psrn) / i_psnr * index
                        # mu = tmp if tmp > torch.ones_like(i_psnr) else torch.ones_like(i_psnr)
                        # print(index, mu)
                        # print(1, mc_loss, mc_w * mc_loss)
                        # print(2, mse_loss + mc_w * mc_loss)
                        # exit()
                        # _mse += (mse_loss + mc_w * mc_loss) * mu
                        _mse += mse_loss * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())
                        _loss = self.l_PSNR * mse_loss + bpp
                        # loss += _loss

                        train_i_psnr.update(i_psnr, self.batch_size)
                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        # self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                        # self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                        # self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                        # self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                        # self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                        # self.writer.add_scalar('train_psnr', psrn, self.global_step)
                        # self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                        # self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                        # self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                        # self.writer.add_scalar('train_loss', _loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T-ALL{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.3f}|{:.3f}|{:.3f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   i_psnr,
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                    num = f * (f - 1) // 2
                    # print(_loss)
                    # print(_loss.div(f - 1), f - 1)
                    # exit()
                    # loss = self.l_PSNR * _mse + _bpp
                    loss = self.l_PSNR * _mse.div(num) + _bpp.div(num)
                    # loss = _loss.div(f - 1)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    _aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 10:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_psnr.avg,
                                    train_warp_psnr.avg,
                                    train_mc_psnr.avg,
                                    train_psnr.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            # if epoch % self.args.val_freq == 0:
            #     self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()
        val_warp_psnr, val_mc_psnr = AverageMeter(), AverageMeter()
        val_res_bpp, val_mv_bpp = AverageMeter(), AverageMeter()
        val_psnr, val_msssim, val_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
        val_res_aux, val_mv_aux, val_aux = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                i_mse = torch.mean((frames[0] - ref_frame).pow(2))
                i_psnr = 10 * np.log10(1.0 / i_mse.detach().cpu())
                smi = self.sm_i(self.process(frames[0].mul(255)))
                self.supp = torch.cat([smi[0], smi[0]], 0)
                f = self.get_f()
                feature = None
                for frame_index in range(1, f):
                    curr_frame = frames[frame_index]
                    sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                    self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                    decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                        self.graph(ref_frame, curr_frame, y, feature=feature)
                    ref_frame = decoded_frame.detach().clone()
                    feature = feature1.detach().clone()
                    loss = self.l_PSNR * mse_loss + bpp

                    msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                    psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                    mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                    warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                    mv_aux = self.graph.mv_aux_loss()
                    res_aux = self.graph.res_aux_loss()
                    aux = self.graph.aux_loss()

                    val_i_psnr.update(i_psnr.mean().detach().item(), self.batch_size)
                    val_loss.update(loss.mean().detach().item(), self.batch_size)
                    val_warp_psnr.update(warp_psrn, self.batch_size)
                    val_mc_psnr.update(mc_psrn, self.batch_size)
                    val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                    val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                    val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                    val_msssim.update(msssim, self.batch_size)
                    val_psnr.update(psrn, self.batch_size)

                    val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                    val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                    val_aux.update(aux.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar('val_mv_aux', mv_aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_res_aux', res_aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_aux', aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_psnr', psrn, self.global_step)
                    self.writer.add_scalar('val_loss', loss.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_warp_psnr', warp_psrn, self.global_eval_step)
                    self.writer.add_scalar('val_mc_psnr', mc_psrn, self.global_eval_step)
                    self.writer.add_scalar('val_bpp', bpp.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_res_bpp', bpp_res.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_mv_bpp', bpp_mv.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_msssim', msssim, self.global_eval_step)
                    self.writer.add_scalar('val_psnr', psrn, self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.2f}], PSNR[{:.2f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                     "AUX[{:.1f}|{:.1f}|{:.1f}]".format(
                        self.global_epoch + 1,
                        self.num_epochs,
                        loss.mean().detach().item(),
                        i_psnr,
                        warp_psrn,
                        mc_psrn,
                        psrn,
                        bpp_mv.mean().detach().item(),
                        bpp_res.mean().detach().item(),
                        bpp.mean().detach().item(),
                        mv_aux.detach().item(),
                        res_aux.detach().item(),
                        aux.detach().item(),
                    )

                self.global_eval_step += 1

                # if k > 10:
                #     break

        self.logger.info("VALID [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                         "AUX[{:.1f}|{:.1f}|{:.1f}]". \
                         format(self.global_epoch + 1,
                                self.num_epochs,
                                val_loss.avg,
                                val_i_psnr.avg,
                                val_warp_psnr.avg,
                                val_mc_psnr.avg,
                                val_psnr.avg,
                                val_mv_bpp.avg,
                                val_res_bpp.avg,
                                val_bpp.avg,
                                val_mv_aux.avg,
                                val_res_aux.avg,
                                val_aux.avg
                                ))
        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 4
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 7
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch  # last_epoch
        # self.global_step = int(self.stage5_step)  # checkpoint["global_step"] + 1
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # if self.global_step < self.stage3_step:
        #     pass
        # elif self.stage3_step < self.global_step <= self.stage4_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        # else:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

        if self.global_step >= int(900000):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

        if self.global_step >= int(960000):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 50.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 50.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = list(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = list(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class Trainer_ICIP2020ResB_WSM_PSNRv2(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)
        self.mc_weights = 0.01

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 3frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        self.grad_clip = 1.0
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM
        lmbda, fea_mse, beta = 0.0, 0.0, 0.0
        if self.l_PSNR == 1280:
            lmbda, fea_mse, beta = 0.0932, 64, 48
        elif self.l_PSNR == 640:
            # lmbda, fea_mse, beta = 0.0483, 32, 24
            # lmbda, fea_mse, beta = 0.0483, 4, 3
            # lmbda, fea_mse, beta = 0.0483, 8, 6
            # lmbda, fea_mse, beta = 0.0483, 16, 12
            lmbda, fea_mse, beta = 0.0483, 20, 16
        elif self.l_PSNR == 320:
            lmbda, fea_mse, beta = 0.025, 16, 12
        elif self.l_PSNR == 160:
            lmbda, fea_mse, beta = 0.013, 8, 6
        elif self.l_PSNR == 80:
            lmbda, fea_mse, beta = 0.0067, 4, 3
        else:
            print('I-lambda error')
            exit()

        # if self.l_PSNR == 1280:
        #     lmbda, fea_mse, beta = 0.0932, 64, 48
        # elif self.l_PSNR == 960:
        #     lmbda, fea_mse, beta = 0.0483, 20, 16
        # elif self.l_PSNR == 480:
        #     lmbda, fea_mse, beta = 0.025, 16, 12
        # elif self.l_PSNR == 160:
        #     lmbda, fea_mse, beta = 0.013, 8, 6
        # elif self.l_PSNR == 80:
        #     lmbda, fea_mse, beta = 0.0067, 3, 3
        # else:
        #     print('I-lambda error')
        #     exit()

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        # self.log_dir = os.path.join(args.log_root, f"15resb_Vimeo_WSM_I{lmbda}_PSNR{args.l_PSNR}_{beta}{fea_mse}_v2_load68bpg_{date}")
        # self.log_dir = os.path.join(args.log_root, f"JointTrain_Vimeo_WSM_I{lmbda}_PSNR{args.l_PSNR}_{beta}{fea_mse}_v2_{date}")
        self.log_dir = os.path.join(args.log_root,
                                    f"LSVC_ablation21_I{lmbda}_PSNR{args.l_PSNR}_{beta}{fea_mse}_v2_{date}")
        # self.log_dir = os.path.join(args.log_root,
        #                             f"322_nores_Vimeo_WSM_I{lmbda}_PSNR{args.l_PSNR}_{beta}{fea_mse}_v2_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size

        training_set, valid_set = get_dataset(args, mf=args.frames, return_orgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = int(self.stage5_step)
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # model
        self.mode_type = args.mode_type
        # self.graph = DVC2_MC_322_15(True).cuda()  DVC2_MC_322_1ablation1  DVC2_MC_322_1  DVC2_MC_322_1_nores
        self.graph = DeepSVC().cuda()
        self.resb15 = False

        # self.logger.info(f"[*] Load Pretrained Model...")
        # ckpt = './checkpoint/LHB_DVC_WOSM_bpg2048.pth'
        # tgt_model_dict = self.graph.state_dict()
        # src_pretrained_dict = torch.load(ckpt)['state_dict']
        # _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        # tgt_model_dict.update(_pretrained_dict)
        # self.graph.load_state_dict(tgt_model_dict)

        self.i_codec = ICIP2020ResB()  # 3, 4, 5, 6
        ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/mse/lambda_{lmbda}.pth'
        self.logger.info(f'[*] Load i_codec from {ckpt}')
        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = load_pretrained(checkpoint["state_dict"])
        self.i_codec.load_state_dict(state_dict)
        self.i_codec.update(force=True)
        self.i_codec.eval()
        self.i_codec.cuda()
        for p in self.i_codec.parameters():
            p.requires_grad = False

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        # restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_i_jointTrain.pth'
        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_i.pth'
        self.logger.info(f'[*] Load sm_i from {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_i = ResNetTeacher()
        self.sm_i.load_state_dict(pcheckpoint["state_dict"])
        self.sm_i.eval()
        self.sm_i.cuda()
        for p in self.sm_i.parameters():
            p.requires_grad = False

        # restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_p_jointTrain.pth'
        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_p.pth'
        self.logger.info(f'[*] Load sm_p from {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_p = OursResNetStudentP(N=72)
        self.sm_p.load_state_dict(pcheckpoint["state_dict"])
        self.sm_p.layer1.update(force=True)
        self.sm_p.eval()
        self.sm_p.cuda()
        for p in self.sm_p.parameters():
            p.requires_grad = False

        self.process = Process()
        self.supp = None

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_psnr, train_mc_psnr = AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp, train_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            # adjust learning_rate
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                self.adjust_lr()
                if self.stage3_step < self.global_step and self.global_step % 5e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                with torch.no_grad():
                    ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                    i_mse = torch.mean((frames[0] - ref_frame).pow(2))
                    i_psnr = 10 * np.log10(1.0 / i_mse.detach().cpu())

                    smi = self.sm_i(self.process(frames[0].mul(255)))
                    self.supp = torch.cat([smi[0], smi[0]], 0)

                f = self.get_f()
                feature = None
                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        with torch.no_grad():
                            sm = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]

                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, sm[0], feature=feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            warp_weight = 0.1
                        else:
                            warp_weight = 0
                        distortion = mse_loss + warp_weight * (warp_loss + mc_loss)
                        loss = self.l_PSNR * distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                        train_i_psnr.update(i_psnr, self.batch_size)
                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T-ALL{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   i_psnr,
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _mse, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        with torch.no_grad():
                            sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                        # ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                        #     self.graph(ref_frame, curr_frame, sm[0], feature=feature)

                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, sm[0], feature=feature)
                        # ref_frame = decoded_frame.detach().clone()
                        psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        # mu = torch.ones_like(i_psnr) + (i_psnr - psrn) / i_psnr * index
                        # _mse += mse_loss * mu
                        _mse += (mse_loss + self.mc_weights * mc_loss) * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())
                        _loss = self.l_PSNR * mse_loss + bpp

                        train_i_psnr.update(i_psnr, self.batch_size)
                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                        self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                        self.writer.add_scalar('train_psnr', psrn, self.global_step)
                        self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_loss', _loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "TALL{} [{}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   # epoch + 1,
                                   # self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   i_psnr,
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                    num = f * (f - 1) // 2
                    loss = self.l_PSNR * _mse.div(num) + _bpp.div(num)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    _aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 10:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_psnr.avg,
                                    train_warp_psnr.avg,
                                    train_mc_psnr.avg,
                                    train_psnr.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            # if epoch % self.args.val_freq == 0:
            #     self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()
        val_warp_psnr, val_mc_psnr = AverageMeter(), AverageMeter()
        val_res_bpp, val_mv_bpp = AverageMeter(), AverageMeter()
        val_psnr, val_msssim, val_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
        val_res_aux, val_mv_aux, val_aux = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                i_mse = torch.mean((frames[0] - ref_frame).pow(2))
                i_psnr = 10 * np.log10(1.0 / i_mse.detach().cpu())
                smi = self.sm_i(self.process(frames[0].mul(255)))
                self.supp = torch.cat([smi[0], smi[0]], 0)
                f = self.get_f()
                feature = None
                for frame_index in range(1, f):
                    curr_frame = frames[frame_index]
                    sm = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                    self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                    decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                        self.graph(ref_frame, curr_frame, sm[0], feature=feature)
                    ref_frame = decoded_frame.detach().clone()
                    feature = feature1.detach().clone()
                    loss = self.l_PSNR * mse_loss + bpp

                    msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                    psrn = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                    mc_psrn = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                    warp_psrn = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                    mv_aux = self.graph.mv_aux_loss()
                    res_aux = self.graph.res_aux_loss()
                    aux = self.graph.aux_loss()

                    val_i_psnr.update(i_psnr.mean().detach().item(), self.batch_size)
                    val_loss.update(loss.mean().detach().item(), self.batch_size)
                    val_warp_psnr.update(warp_psrn, self.batch_size)
                    val_mc_psnr.update(mc_psrn, self.batch_size)
                    val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                    val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                    val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                    val_msssim.update(msssim, self.batch_size)
                    val_psnr.update(psrn, self.batch_size)

                    val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                    val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                    val_aux.update(aux.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar('val_mv_aux', mv_aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_res_aux', res_aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_aux', aux.detach().item(), self.global_step)
                    self.writer.add_scalar('val_psnr', psrn, self.global_step)
                    self.writer.add_scalar('val_loss', loss.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_warp_psnr', warp_psrn, self.global_eval_step)
                    self.writer.add_scalar('val_mc_psnr', mc_psrn, self.global_eval_step)
                    self.writer.add_scalar('val_bpp', bpp.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_res_bpp', bpp_res.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_mv_bpp', bpp_mv.mean().detach().item(), self.global_eval_step)
                    self.writer.add_scalar('val_msssim', msssim, self.global_eval_step)
                    self.writer.add_scalar('val_psnr', psrn, self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.2f}], PSNR[{:.2f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                     "AUX[{:.1f}|{:.1f}|{:.1f}]".format(
                        self.global_epoch + 1,
                        self.num_epochs,
                        loss.mean().detach().item(),
                        i_psnr,
                        warp_psrn,
                        mc_psrn,
                        psrn,
                        bpp_mv.mean().detach().item(),
                        bpp_res.mean().detach().item(),
                        bpp.mean().detach().item(),
                        mv_aux.detach().item(),
                        res_aux.detach().item(),
                        aux.detach().item(),
                    )

                self.global_eval_step += 1

                # if k > 10:
                #     break

        self.logger.info("VALID [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                         "AUX[{:.1f}|{:.1f}|{:.1f}]". \
                         format(self.global_epoch + 1,
                                self.num_epochs,
                                val_loss.avg,
                                val_i_psnr.avg,
                                val_warp_psnr.avg,
                                val_mc_psnr.avg,
                                val_psnr.avg,
                                val_mv_bpp.avg,
                                val_res_bpp.avg,
                                val_bpp.avg,
                                val_mv_aux.avg,
                                val_res_aux.avg,
                                val_aux.avg
                                ))
        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 4
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 7
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch  # last_epoch
        self.global_step = int(self.stage5_step)  # checkpoint["global_step"] + 1
        # self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # if self.global_step < self.stage3_step:
        #     pass
        # elif self.stage3_step < self.global_step <= self.stage4_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        # else:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

        # 1020000
        # print(self.global_step, self.global_step > int(950000 - 2))
        # 1035000  760000   890000  900000
        # self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        # self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        if self.global_step > int(self.stage5_step - 2):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            # print(1)
        if self.global_step > int(self.stage5_step + 30000):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
            if self.global_step > int(self.stage5_step + 50000):
                self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
                self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
            # print(11)

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        # if is_best:
        #     torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = list(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = list(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class Trainer_ICIP2020ResB_WSM_MSSSIM(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 3frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        # self.stage1_step = 30  # 2frames
        # self.stage2_step = self.stage1_step + 30  # 3frames
        # self.stage3_step = self.stage2_step + 30  # 4frames
        # self.stage4_step = self.stage3_step + 30  # 5frames
        # self.stage5_step = self.stage4_step + 30  # 5frames

        self.grad_clip = 1.0
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_PSNR / 50  # TCM50  DCVC32  Compressai50
        lmbda, fea_mse, beta = 0.0, 0.0, 0.0
        if self.l_PSNR == 640:
            # lmbda, fea_mse, beta = 60.50, 32, 24
            lmbda, fea_mse, beta = 60.5, 20, 16
        elif self.l_PSNR == 320:
            lmbda, fea_mse, beta = 31.73, 16, 12
        elif self.l_PSNR == 160:
            lmbda, fea_mse, beta = 16.64, 8, 6
        elif self.l_PSNR == 80:
            lmbda, fea_mse, beta = 8.73, 4, 3
        else:
            print('I-lambda error')
            exit()

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"15resb_Vimeo_WSM_I{lmbda}_MSSSIM_{beta}{fea_mse}_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')
        self.logger.info(f'[*] MS-SSIM Lambda = {self.l_MSSSIM}')

        # data
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size

        training_set, valid_set = get_dataset(args, mf=5, return_orgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = int(self.stage5_step)
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # model
        self.mode_type = args.mode_type
        self.graph = DeepSVC().cuda()

        self.i_codec = ICIP2020ResB()  # 3, 4, 5, 6
        ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/msssim_from0/lambda_{lmbda}.pth'
        self.logger.info(f'[*] Load i_codec {ckpt}')
        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = load_pretrained(checkpoint["state_dict"])
        self.i_codec.load_state_dict(state_dict)
        self.i_codec.update(force=True)
        self.i_codec.eval()
        self.i_codec.cuda()
        for p in self.i_codec.parameters():
            p.requires_grad = False

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_i.pth'
        self.logger.info(f'[*] Load smi {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_i = ResNetTeacher()
        self.sm_i.load_state_dict(pcheckpoint["state_dict"])
        self.sm_i.eval()
        self.sm_i.cuda()
        for p in self.sm_i.parameters():
            p.requires_grad = False

        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_p.pth'
        self.logger.info(f'[*] Load smp {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_p = OursResNetStudentP(N=72)
        self.sm_p.load_state_dict(pcheckpoint["state_dict"])
        self.sm_p.layer1.update(force=True)
        self.sm_p.eval()
        self.sm_p.cuda()
        for p in self.sm_p.parameters():
            p.requires_grad = False

        self.process = Process()
        self.supp = None

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        # self.validate()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_msssim, train_mc_msssim, train_msssim = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp, train_i_msssim = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            # adjust learning_rate
            # self.adjust_lr()
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                self.adjust_lr()
                if self.stage3_step < self.global_step and self.global_step % 5e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                with torch.no_grad():
                    ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                    i_msssim = ms_ssim(ref_frame, frames[0], 1.0)
                    smi = self.sm_i(self.process(frames[0].mul(255)))
                    self.supp = torch.cat([smi[0], smi[0]], 0)

                f = self.get_f()
                feature = None
                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        with torch.no_grad():
                            sm = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]

                        decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, sm[0], feature=feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            warp_weight = 0.2
                        else:
                            warp_weight = 0
                        distortion = (1 - msssim) + warp_weight * (2 - warp_msssim - mc_msssim)
                        loss = self.l_MSSSIM * distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        train_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        train_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        train_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        train_msssim.update(msssim.mean().detach().item(), self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)

                        train_bar.desc = "T-ALL{} [{}|{}|{}] LOSS[{:.2f}], MS-SSIM[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   i_msssim.mean().detach().item(),
                                   warp_msssim.mean().detach().item(),
                                   mc_msssim.mean().detach().item(),
                                   msssim.mean().detach().item(),
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _msssim, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        with torch.no_grad():
                            sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                        ref_frame, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, y, feature=feature)
                        # mu = torch.ones_like(i_msssim.mean().detach()) + \
                        #      (i_msssim.mean().detach() - msssim.mean().detach()) / i_msssim.mean().detach() * index
                        _msssim += (1 - msssim) * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss()

                        # print(msssim)
                        # print(warp_msssim)
                        # print(mc_msssim)
                        # exit()

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        _loss = self.l_MSSSIM * (1 - msssim) + bpp

                        train_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        train_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        train_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        train_msssim.update(msssim.mean().detach().item(), self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)

                        train_bar.desc = "Final{} [{}] LOSS[{:.1f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   # epoch + 1,
                                   # self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   i_msssim.mean().detach().item(),
                                   warp_msssim.mean().detach().item(),
                                   mc_msssim.mean().detach().item(),
                                   msssim.mean().detach().item(),
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                    num = f * (f - 1) // 2
                    loss = self.l_MSSSIM * _msssim.div(num) + _bpp.div(num)
                    # loss = self.l_MSSSIM * _msssim + _bpp

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    _aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 10:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.4f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_msssim.avg,
                                    train_warp_msssim.avg,
                                    train_mc_msssim.avg,
                                    train_msssim.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            if epoch % self.args.val_freq == 0:
                self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()
        val_warp_msssim, val_mc_msssim = AverageMeter(), AverageMeter()
        val_res_bpp, val_mv_bpp = AverageMeter(), AverageMeter()
        val_msssim, val_i_msssim = AverageMeter(), AverageMeter()
        val_res_aux, val_mv_aux, val_aux = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                i_msssim = ms_ssim(ref_frame, frames[0], 1.0)
                smi = self.sm_i(self.process(frames[0].mul(255)))
                self.supp = torch.cat([smi[0], smi[0]], 0)
                f = self.get_f()
                feature = None
                for frame_index in range(1, f):
                    curr_frame = frames[frame_index]
                    sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                    self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                    decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                        self.graph.forward_msssim(ref_frame, curr_frame, y, feature=feature)
                    ref_frame = decoded_frame.detach().clone()
                    feature = feature1.detach().clone()
                    loss = self.l_MSSSIM * (1 - msssim) + bpp
                    mv_aux = self.graph.mv_aux_loss()
                    res_aux = self.graph.res_aux_loss()
                    aux = self.graph.aux_loss()

                    val_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                    val_loss.update(loss.mean().detach().item(), self.batch_size)
                    val_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                    val_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                    val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                    val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                    val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                    val_msssim.update(msssim.mean().detach().item(), self.batch_size)

                    val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                    val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                    val_aux.update(aux.mean().detach().item(), self.batch_size)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.1f}], MS-SSIM[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                     "BPP[{:.3f}|{:.3f}|{:.3f}] AUX[{:.1f}|{:.1f}|{:.1f}]".format(
                        self.global_epoch + 1,
                        self.num_epochs,
                        loss.mean().detach().item(),
                        i_msssim.mean().detach().item(),
                        warp_msssim.mean().detach().item(),
                        mc_msssim.mean().detach().item(),
                        msssim.mean().detach().item(),
                        bpp_mv.mean().detach().item(),
                        bpp_res.mean().detach().item(),
                        bpp.mean().detach().item(),
                        mv_aux.detach().item(),
                        res_aux.detach().item(),
                        aux.detach().item(),
                    )

                self.global_eval_step += 1

                # if k > 10:
                #     break

        self.logger.info("VALID [{}|{}] LOSS[{:.4f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                         "AUX[{:.1f}|{:.1f}|{:.1f}]". \
                         format(self.global_epoch + 1,
                                self.num_epochs,
                                val_loss.avg,
                                val_i_msssim.avg,
                                val_warp_msssim.avg,
                                val_mc_msssim.avg,
                                val_msssim.avg,
                                val_mv_bpp.avg,
                                val_res_bpp.avg,
                                val_bpp.avg,
                                val_mv_aux.avg,
                                val_res_aux.avg,
                                val_aux.avg
                                ))
        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 4
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 7
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch  # last_epoch
        # self.global_step = int(self.stage5_step)  # checkpoint["global_step"] + 1
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # if self.global_step < self.stage3_step:
        #     pass
        # elif self.stage3_step < self.global_step <= self.stage4_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        # else:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

        if self.global_step >= int(900000):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 4.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 4.0

        if self.global_step >= int(980000):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

        if self.global_step >= int(1020000):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 50.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 50.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = list(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = list(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class Trainer_ICIP2020ResB_WSM_MSSSIMv2(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)
        self.mc_weights = 0.01

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 3frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        # self.stage1_step = 30  # 2frames
        # self.stage2_step = self.stage1_step + 30  # 3frames
        # self.stage3_step = self.stage2_step + 30  # 4frames
        # self.stage4_step = self.stage3_step + 30  # 5frames
        # self.stage5_step = self.stage4_step + 30  # 5frames

        self.grad_clip = 1.0
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_PSNR / 50  # TCM50  DCVC32  Compressai50
        lmbda, fea_mse, beta = 0.0, 0.0, 0.0
        if self.l_PSNR == 640:
            # lmbda, fea_mse, beta = 60.50, 32, 24
            lmbda, fea_mse, beta = 60.50, 20, 16
        elif self.l_PSNR == 320:
            lmbda, fea_mse, beta = 31.73, 16, 12
        elif self.l_PSNR == 160:
            lmbda, fea_mse, beta = 16.64, 8, 6
        elif self.l_PSNR == 80:
            # lmbda, fea_mse, beta = 8.73, 3, 3
            lmbda, fea_mse, beta = 8.73, 4, 3
        else:
            print('I-lambda error')
            exit()

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        # self.log_dir = os.path.join(args.log_root, f"JointTrain_Vimeo_WSM_I{lmbda}_MSSSIM_{beta}{fea_mse}_v2_{date}")
        # self.log_dir = os.path.join(args.log_root, f"Vimeo_WSM_I{lmbda}_MSSSIM_{beta}{fea_mse}_v2_{date}")
        self.log_dir = os.path.join(args.log_root, f"LSVC_I{lmbda}_MSSSIM_{beta}{fea_mse}_v2_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')
        self.logger.info(f'[*] MS-SSIM Lambda = {self.l_MSSSIM}')

        # data
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size

        training_set, valid_set = get_dataset(args, mf=5, return_orgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = int(self.stage5_step)
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # model
        self.mode_type = args.mode_type
        self.graph = DeepSVC().cuda()

        self.i_codec = ICIP2020ResB()  # 3, 4, 5, 6
        ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/msssim_from0/lambda_{lmbda}.pth'
        self.logger.info(f'[*] Load i_codec {ckpt}')
        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = load_pretrained(checkpoint["state_dict"])
        self.i_codec.load_state_dict(state_dict)
        self.i_codec.update(force=True)
        self.i_codec.eval()
        self.i_codec.cuda()
        for p in self.i_codec.parameters():
            p.requires_grad = False

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_i.pth'
        self.logger.info(f'[*] Load smi {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_i = ResNetTeacher()
        self.sm_i.load_state_dict(pcheckpoint["state_dict"])
        self.sm_i.eval()
        self.sm_i.cuda()
        for p in self.sm_i.parameters():
            p.requires_grad = False

        restore_path = f'./checkpoints/baselayer/beta{beta}_mse{fea_mse}_p.pth'
        self.logger.info(f'[*] Load smp {restore_path}')
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        self.sm_p = OursResNetStudentP(N=72)
        self.sm_p.load_state_dict(pcheckpoint["state_dict"])
        self.sm_p.layer1.update(force=True)
        self.sm_p.eval()
        self.sm_p.cuda()
        for p in self.sm_p.parameters():
            p.requires_grad = False

        self.process = Process()
        self.supp = None

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        # self.validate()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_msssim, train_mc_msssim, train_msssim = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp, train_i_msssim = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            # adjust learning_rate
            # self.adjust_lr()
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                self.adjust_lr()
                if self.stage3_step < self.global_step and self.global_step % 5e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                with torch.no_grad():
                    ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                    i_msssim = ms_ssim(ref_frame, frames[0], 1.0)
                    smi = self.sm_i(self.process(frames[0].mul(255)))
                    self.supp = torch.cat([smi[0], smi[0]], 0)

                f = self.get_f()
                feature = None
                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        with torch.no_grad():
                            sm = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]

                        decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, sm[0], feature=feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            warp_weight = 0.2
                        else:
                            warp_weight = 0
                        distortion = (1 - msssim) + warp_weight * (2 - warp_msssim - mc_msssim)
                        loss = self.l_MSSSIM * distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        train_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        train_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        train_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        train_msssim.update(msssim.mean().detach().item(), self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)

                        train_bar.desc = "T-ALL{} [{}|{}|{}] LOSS[{:.2f}], MS-SSIM[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   i_msssim.mean().detach().item(),
                                   warp_msssim.mean().detach().item(),
                                   mc_msssim.mean().detach().item(),
                                   msssim.mean().detach().item(),
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _msssim, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        with torch.no_grad():
                            sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                        # ref_frame, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                        #     self.graph.forward_msssim(ref_frame, curr_frame, y, feature=feature)
                        ref_frame, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, sm[0], feature=feature)
                        # mu = torch.ones_like(i_msssim.mean().detach()) + \
                        #      (i_msssim.mean().detach() - msssim.mean().detach()) / i_msssim.mean().detach() * index
                        # _msssim += (1 - msssim) * mu
                        _msssim += ((1 - msssim) + self.mc_weights * (1 - mc_msssim)) * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss()

                        # print(msssim)
                        # print(warp_msssim)
                        # print(mc_msssim)
                        # exit()

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        _loss = self.l_MSSSIM * (1 - msssim) + bpp

                        train_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        train_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        train_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        train_msssim.update(msssim.mean().detach().item(), self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)

                        train_bar.desc = "Final{} [{}] LOSS[{:.2f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   # epoch + 1,
                                   # self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   i_msssim.mean().detach().item(),
                                   warp_msssim.mean().detach().item(),
                                   mc_msssim.mean().detach().item(),
                                   msssim.mean().detach().item(),
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                    num = f * (f - 1) // 2
                    loss = self.l_MSSSIM * _msssim.div(num) + _bpp.div(num)
                    # loss = self.l_MSSSIM * _msssim + _bpp
                    # loss = self.l_MSSSIM * _msssim + _bpp

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    _aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 10:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.4f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_msssim.avg,
                                    train_warp_msssim.avg,
                                    train_mc_msssim.avg,
                                    train_msssim.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            if epoch % self.args.val_freq == 0:
                self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()
        val_warp_msssim, val_mc_msssim = AverageMeter(), AverageMeter()
        val_res_bpp, val_mv_bpp = AverageMeter(), AverageMeter()
        val_msssim, val_i_msssim = AverageMeter(), AverageMeter()
        val_res_aux, val_mv_aux, val_aux = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = self.i_codec(frames[0])['x_hat'].clamp(0.0, 1.0)
                i_msssim = ms_ssim(ref_frame, frames[0], 1.0)
                smi = self.sm_i(self.process(frames[0].mul(255)))
                self.supp = torch.cat([smi[0], smi[0]], 0)
                f = self.get_f()
                feature = None
                for frame_index in range(1, f):
                    curr_frame = frames[frame_index]
                    sm, y = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                    self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]
                    decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                        self.graph.forward_msssim(ref_frame, curr_frame, sm[0], feature=feature)
                    # decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                    #     self.graph.forward_msssim(ref_frame, curr_frame, y, feature=feature)
                    ref_frame = decoded_frame.detach().clone()
                    feature = feature1.detach().clone()
                    loss = self.l_MSSSIM * (1 - msssim) + bpp
                    mv_aux = self.graph.mv_aux_loss()
                    res_aux = self.graph.res_aux_loss()
                    aux = self.graph.aux_loss()

                    val_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                    val_loss.update(loss.mean().detach().item(), self.batch_size)
                    val_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                    val_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                    val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                    val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                    val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                    val_msssim.update(msssim.mean().detach().item(), self.batch_size)

                    val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                    val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                    val_aux.update(aux.mean().detach().item(), self.batch_size)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.1f}], MS-SSIM[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                     "BPP[{:.3f}|{:.3f}|{:.3f}] AUX[{:.1f}|{:.1f}|{:.1f}]".format(
                        self.global_epoch + 1,
                        self.num_epochs,
                        loss.mean().detach().item(),
                        i_msssim.mean().detach().item(),
                        warp_msssim.mean().detach().item(),
                        mc_msssim.mean().detach().item(),
                        msssim.mean().detach().item(),
                        bpp_mv.mean().detach().item(),
                        bpp_res.mean().detach().item(),
                        bpp.mean().detach().item(),
                        mv_aux.detach().item(),
                        res_aux.detach().item(),
                        aux.detach().item(),
                    )

                self.global_eval_step += 1

                # if k > 1000:
                #     break

        self.logger.info("VALID [{}|{}] LOSS[{:.4f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                         "AUX[{:.1f}|{:.1f}|{:.1f}]". \
                         format(self.global_epoch + 1,
                                self.num_epochs,
                                val_loss.avg,
                                val_i_msssim.avg,
                                val_warp_msssim.avg,
                                val_mc_msssim.avg,
                                val_msssim.avg,
                                val_mv_bpp.avg,
                                val_res_bpp.avg,
                                val_bpp.avg,
                                val_mv_aux.avg,
                                val_res_aux.avg,
                                val_aux.avg
                                ))
        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 4
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 7
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch  # last_epoch
        # self.global_step = int(self.stage5_step)  # checkpoint["global_step"] + 1
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # if self.global_step < self.stage3_step:
        #     pass
        # elif self.stage3_step < self.global_step <= self.stage4_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        # else:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        # 890000  1035000  step_995000
        self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        # if self.global_step >= int(840000):
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        #
        # if self.global_step >= int(840000 + 90000):
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = list(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = list(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

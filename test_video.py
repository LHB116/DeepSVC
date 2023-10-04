# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import time
import numpy as np
from pytorch_msssim import ms_ssim
from image_model import ICIP2020ResB
from semantic_layer import OursResNetStudentP, ResNetTeacher
from video_model import DeepSVC
from modules import read_image, cal_psnr, crop, pad
import json
import glob

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

TEST_DATA = {
    'HEVC_B': {
        'path': '/tdx/LHB/data/TestSets/ClassB',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': [
            'BasketballDrive_1920x1080_50',
            'BQTerrace_1920x1080_60',
            'Cactus_1920x1080_50',
            'Kimono1_1920x1080_24',
            'ParkScene_1920x1080_24',
        ],
    },

    'HEVC_C': {
        'path': '/tdx/LHB/data/TestSets/ClassC/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'BasketballDrill_832x480_50',
            'BQMall_832x480_60',
            'PartyScene_832x480_50',
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_D': {
        'path': '/tdx/LHB/data/TestSets/ClassD/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '416x240',
        'x64_resolution': '384x192',
        'sequences': [
            'BasketballPass_416x240_50',
            'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60',
            'RaceHorses_416x240_30',
        ],
    },

    'HEVC_E': {
        'path': '/tdx/LHB/data/TestSets/ClassE/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1280x720',
        'x64_resolution': '1280x704',
        'sequences': [
            'FourPeople_1280x720_60',
            'Johnny_1280x720_60',
            'KristenAndSara_1280x720_60',
        ],
    },

    'UVG': {
        'path': '/tdx/LHB/data/TestSets/UVG/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': [
            'Beauty_1920x1080_120fps_420_8bit_YUV',
            'Bosphorus_1920x1080_120fps_420_8bit_YUV',
            'HoneyBee_1920x1080_120fps_420_8bit_YUV',
            'Jockey_1920x1080_120fps_420_8bit_YUV',
            'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV',
            'ShakeNDry_1920x1080_120fps_420_8bit_YUV',
            'YachtRide_1920x1080_120fps_420_8bit_YUV',
        ],
    },

    'VTL': {
        'path': '/tdx/LHB/data/ACM23/VTL',
        'frames': 96,
        'gop': 12,
        'org_resolution': '352x288',
        'x64_resolution': '352x288',
        'sequences': [
            'akiyo_cif',
            'BigBuckBunny_CIF_24fps',
            'bridge-close_cif',
            'bridge-far_cif',
            'bus_cif',
            'coastguard_cif',
            'container_cif',
            'ElephantsDream_CIF_24fps',
            'flower_cif',
            'foreman_cif',
            'hall_cif',
            'highway_cif',
            'mobile_cif',
            'mother-daughter_cif',
            'news_cif',
            'paris_cif',
            'silent_cif',
            'stefan_cif',
            'tempete_cif',
            'waterfall_cif',
        ],
    },

    "MCL-JCV": {
        "path": "/tdx/LHB/data/TestSets/MCL-JCV",
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',  # 18,20,24,25
        "sequences": [
            "videoSRC01_1920x1080_30",
            "videoSRC02_1920x1080_30",
            "videoSRC03_1920x1080_30",
            "videoSRC04_1920x1080_30",
            "videoSRC05_1920x1080_25",
            "videoSRC06_1920x1080_25",
            "videoSRC07_1920x1080_25",
            "videoSRC08_1920x1080_25",
            "videoSRC09_1920x1080_25",
            "videoSRC10_1920x1080_30",
            "videoSRC11_1920x1080_30",
            "videoSRC12_1920x1080_30",
            "videoSRC13_1920x1080_30",
            "videoSRC14_1920x1080_30",
            "videoSRC15_1920x1080_30",
            "videoSRC16_1920x1080_30",
            "videoSRC17_1920x1080_24",
            "videoSRC18_1920x1080_25",
            "videoSRC19_1920x1080_30",
            "videoSRC20_1920x1080_25",
            "videoSRC21_1920x1080_24",
            "videoSRC22_1920x1080_24",
            "videoSRC23_1920x1080_24",
            "videoSRC24_1920x1080_24",
            "videoSRC25_1920x1080_24",
            "videoSRC26_1920x1080_30",
            "videoSRC27_1920x1080_30",
            "videoSRC28_1920x1080_30",
            "videoSRC29_1920x1080_24",
            "videoSRC30_1920x1080_30",
        ]
    }
}


def get_parameters(l_PSNR=-1):
    I_lamdba_p, I_lamdba_m, lamdba1, beta1 = 0.0, 0.0, 0, 0
    if l_PSNR == 80:
        I_lamdba_p, I_lamdba_m, lamdba1, beta1 = 0.0067, 8.73, 4, 3
    elif l_PSNR == 160:
        I_lamdba_p, I_lamdba_m, lamdba1, beta1 = 0.013, 16.64, 8, 6
    elif l_PSNR == 320:
        I_lamdba_p, I_lamdba_m, lamdba1, beta1 = 0.025, 31.73, 16, 12
    elif l_PSNR == 640:
        I_lamdba_p, I_lamdba_m, lamdba1, beta1 = 0.0483, 60.5, 20, 16
    return I_lamdba_p, I_lamdba_m, lamdba1, beta1


class Process(torch.nn.Module):
    def __init__(self):
        super(Process, self).__init__()

    def forward(self, tenInput, inverse=False):
        if not inverse:
            tenBlue = (tenInput[:, 0:1, :, :] - 123.675) / 58.395
            tenGreen = (tenInput[:, 1:2, :, :] - 116.28) / 57.12
            tenRed = (tenInput[:, 2:3, :, :] - 103.53) / 57.375
        else:
            tenBlue = tenInput[:, 0:1, :, :] * 58.395 + 123.675
            tenGreen = tenInput[:, 1:2, :, :] * 57.12 + 116.28
            tenRed = tenInput[:, 2:3, :, :] * 57.375 + 103.53
        return torch.cat([tenRed, tenGreen, tenBlue], 1)


# !!!! note that !!!!
# put i frame mode weights in ./checkpoints/ICIP2020ResB
# put Semantic Layer mode weights in ./checkpoints/Semantic_Layer
# put Structure && Texture Layer mode weights in ./checkpoints/PSNR_Final or ./checkpoints/MS_SSIM_Final
def get_result_of_DeepSVC(indicator, test_tgt):
    device = 'cuda:0'
    process = Process()
    test_info = TEST_DATA[test_tgt]
    GOP = test_info['gop']
    total_frame_num = test_info['frames']
    resolution_tgt = 'x64_resolution'
    print(f'Test {test_tgt}')
    print(json.dumps(test_info, indent=2))

    result_save_path = f'./output/testing/{indicator.upper()}/{test_tgt}'
    os.makedirs(result_save_path, exist_ok=True)

    porposed_psnr, porposed_bpp, porposed_msssim = [], [], []
    porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
    porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
    porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
    porposed_mcmsssim, porposed_warmsssim = [], []
    porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
    porposed_ent, porposed_dec, porposed_pbpp_wosm = [], [], []
    porposed_smbpp, porposed_sment, porposed_smdec, porposed_bpp2l = [], [], [], []
    with torch.no_grad():
        for lambda2 in [80, 160, 320, 640]:
            I_lamdba_p, I_lamdba_m, lamdba1, beta1 = get_parameters(lambda2)
            I_lambda = I_lamdba_p if indicator == 'mse' else I_lamdba_m

            if indicator == 'mse':
                log_txt = open(f'{result_save_path}/{lambda2}.txt', 'w')
                restore_path = f'./checkpoints/PSNR_Final/{lambda2}.pth'
            else:
                log_txt = open(f'{result_save_path}/{lambda2}.txt', 'w')
                restore_path = f'./checkpoints/MS_SSIM_Final/{lambda2}.pth'

            p_model = DeepSVC()
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            print(f"INFO Load Pretrained Structure && Texture Layer Model From {restore_path}...")
            p_model.load_state_dict(pcheckpoint["state_dict"])
            p_model.eval()
            p_model.update(force=True)
            p_model.to(device)

            restore_path = f'./checkpoints/Semantic_Layer/beta{beta1}_mse{lamdba1}_i.pth'
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            sm_i = ResNetTeacher()
            sm_i.load_state_dict(pcheckpoint["state_dict"])
            sm_i.eval()
            sm_i.cuda()

            restore_path = f'./checkpoints/Semantic_Layer/beta{beta1}_mse{lamdba1}_p.pth'
            print(f"INFO Load Pretrained Semantic Layer Model From {restore_path}...")
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            sm_p = OursResNetStudentP(N=72)
            sm_p.load_state_dict(pcheckpoint["state_dict"])
            sm_p.layer1.update(force=True)
            sm_p.eval()
            sm_p.cuda()

            i_model = ICIP2020ResB()
            if indicator == 'mse':
                i_restore_path = f'./checkpoints/ICIP2020ResB/{indicator}/lambda_{I_lambda}.pth'
            else:
                i_restore_path = f'./checkpoints/ICIP2020ResB/{indicator}/lambda_{I_lambda}.pth'
            icheckpoint = torch.load(i_restore_path, map_location='cpu')
            print(f"INFO Load Pretrained I-Model From {i_restore_path}...")
            # state_dict = load_pretrained(icheckpoint["state_dict"])
            i_model.load_state_dict(icheckpoint["state_dict"])
            i_model.update(force=True)
            i_model.to(device)
            i_model.eval()

            PSNR, MSSSIM, Bits, Bitswosm = [], [], [], []
            iPSNR, iMSSSIM, iBits = [], [], []
            pPSNR, pMSSSIM, pBits = [], [], []
            mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
            mcMSSSIM, warpMSSSIM = [], []
            iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
            smBits, smEnc, smDec, Bits2l = [], [], [], []
            for ii, seq_info in enumerate(test_info['sequences']):
                _PSNR, _MSSSIM, _Bits, _Bitswosm = [], [], [], []
                _iPSNR, _iMSSSIM, _iBits = [], [], []
                _pPSNR, _pMSSSIM, _pBits = [], [], []
                _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                _mcMSSSIM, _warpMSSSIM = [], []
                _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []
                _smBits, _smEnc, _smDec, _Bits2l = [], [], [], []
                video_frame_path = os.path.join(test_info['path'], 'PNG_Frames',
                                                seq_info.replace(test_info['org_resolution'],
                                                                 test_info[resolution_tgt]))
                images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                image = read_image(images[0]).unsqueeze(0)
                _, _, org_h, org_w = image.size()
                feature = None
                for i, im in enumerate(images):
                    if i >= total_frame_num:
                        break
                    curr_frame_org = read_image(im).unsqueeze(0).to(device)
                    curr_frame = pad(curr_frame_org, 64)
                    num_pixels = curr_frame_org.size(0) * curr_frame_org.size(2) * curr_frame_org.size(3)
                    if i % GOP == 0:
                        feature = None
                        torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        i_out_enc = i_model.compress(curr_frame)
                        torch.cuda.synchronize()
                        elapsed_enc = time.perf_counter() - start_time
                        torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"])
                        torch.cuda.synchronize()
                        elapsed_dec = time.perf_counter() - start_time

                        i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                        i_psnr = cal_psnr(curr_frame_org, crop(i_out_dec["x_hat"], (org_h, org_w)))
                        i_ms_ssim = ms_ssim(curr_frame_org, crop(i_out_dec["x_hat"], (org_h, org_w)),
                                            data_range=1.0).item()

                        _iPSNR.append(i_psnr)
                        _iMSSSIM.append(i_ms_ssim)
                        _iBits.append(i_bpp)
                        _Bitswosm.append(i_bpp)
                        _PSNR.append(i_psnr)
                        _MSSSIM.append(i_ms_ssim)
                        _Bits.append(i_bpp)
                        _iEnc.append(elapsed_enc)
                        _iDec.append(elapsed_dec)
                        _Enc.append(elapsed_enc)
                        _Dec.append(elapsed_dec)
                        _Bits2l.append(i_bpp)
                        print(
                            f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                        log_txt.write(
                            f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                        log_txt.flush()

                        ref_frame = i_out_dec["x_hat"]
                        smi = sm_i(process(ref_frame.mul(255)))
                        supp = torch.cat([smi[0], smi[0]], 0)
                    else:
                        sm, y = sm_p(process(curr_frame.mul(255)), supp, mean=True, encode=True)
                        sm_bpp = sm_p.bpp_loss
                        supp = torch.cat([supp, sm[0]], 0)[1:]

                        torch.cuda.synchronize()
                        start = time.time()
                        mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame, sm[0], feature)
                        torch.cuda.synchronize()
                        elapsed_enc = time.time() - start

                        torch.cuda.synchronize()
                        start = time.time()
                        feature1, dec_p_frame, warped_frame, predict_frame = \
                            p_model.decompress(ref_frame, mv_out_enc, res_out_enc, sm[0], feature)
                        torch.cuda.synchronize()
                        elapsed_dec = time.time() - start

                        mse = torch.mean((curr_frame_org - crop(dec_p_frame, (org_h, org_w))).pow(2)).item()
                        p_psnr = 10 * np.log10(1.0 / mse).item()
                        w_mse = torch.mean((curr_frame_org - crop(warped_frame, (org_h, org_w))).pow(2)).item()
                        w_psnr = 10 * np.log10(1.0 / w_mse).item()
                        mc_mse = torch.mean((curr_frame_org - crop(predict_frame, (org_h, org_w))).pow(2)).item()
                        mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                        p_ms_ssim = ms_ssim(curr_frame_org, crop(dec_p_frame, (org_h, org_w)), data_range=1.0).item()
                        p_warp_ms_ssim = ms_ssim(curr_frame_org, crop(warped_frame, (org_h, org_w)),
                                                 data_range=1.0).item()
                        p_mc_ms_ssim = ms_ssim(curr_frame_org, crop(predict_frame, (org_h, org_w)),
                                               data_range=1.0).item()
                        res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                        mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                        p_bpp = mv_bpp + res_bpp + sm_bpp

                        ref_frame = dec_p_frame.detach()
                        feature = feature1.detach()

                        _PSNR.append(p_psnr)
                        _MSSSIM.append(p_ms_ssim)
                        _Bits.append(p_bpp)
                        _Bitswosm.append(mv_bpp + res_bpp)
                        _pPSNR.append(p_psnr)
                        _pMSSSIM.append(p_ms_ssim)
                        _pBits.append(p_bpp)

                        _mcPSNR.append(mc_psnr)
                        _warpPSNR.append(w_psnr)
                        _mcMSSSIM.append(p_mc_ms_ssim)
                        _warpMSSSIM.append(p_warp_ms_ssim)
                        _mvBits.append(mv_bpp)
                        _resBits.append(res_bpp)
                        _Bits2l.append(mv_bpp + sm_bpp)
                        _smBits.append(sm_bpp)

                        _smEnc.append(sm_p.enct)
                        _smDec.append(sm_p.dect)
                        _pEnc.append(elapsed_enc + sm_p.enct)
                        _pDec.append(elapsed_dec + sm_p.dect)
                        _Enc.append(elapsed_enc + sm_p.enct)
                        _Dec.append(elapsed_dec + sm_p.dect)
                        print(
                            f"i={i}, {seq_info} P-Frame | bpp [{sm_bpp:.3f}, {mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                            f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                            f"| MS-SSIM [{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}|{p_ms_ssim:.3f}] "
                            f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                        log_txt.write(
                            f"i={i}, {seq_info} P-Frame | bpp [{sm_bpp:.3f}, {mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                            f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                            f"| MS-SSIM MS-SSIM [{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}|{p_ms_ssim:.3f}] "
                            f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                        log_txt.flush()

                print(f'I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                print(f'P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                print(f'Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                log_txt.write(f'I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                log_txt.write(f'P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                log_txt.write(f'Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')
                PSNR.append(np.average(_PSNR))
                MSSSIM.append(np.average(_MSSSIM))
                Bits.append(np.average(_Bits))
                Bitswosm.append(np.average(_Bitswosm))

                iPSNR.append(np.average(_iPSNR))
                iMSSSIM.append(np.average(_iMSSSIM))
                iBits.append(np.average(_iBits))
                pPSNR.append(np.average(_pPSNR))
                pMSSSIM.append(np.average(_pMSSSIM))
                pBits.append(np.average(_pBits))
                mcPSNR.append(np.average(_mcPSNR))
                warpPSNR.append(np.average(_warpPSNR))
                mvBits.append(np.average(_mvBits))
                resBits.append(np.average(_resBits))
                mcMSSSIM.append(np.average(_mcMSSSIM))
                warpMSSSIM.append(np.average(_warpMSSSIM))
                iEnc.append(np.average(_iEnc))
                iDec.append(np.average(_iDec))
                pEnc.append(np.average(_pEnc))
                pDec.append(np.average(_pDec))
                Enc.append(np.average(_Enc))
                Dec.append(np.average(_Dec))

                smBits.append(np.average(_smBits))
                smEnc.append(np.average(_smEnc))
                smDec.append(np.average(_smDec))
                Bits2l.append(np.average(_Bits2l))

            results = {
                "psnr": PSNR, "bpp": Bits, "msssim": MSSSIM,
                "ipsnr": iPSNR, "ibpp": iBits, "imsssim": iMSSSIM,
                "ppsnr": pPSNR, "pbpp": pBits, "porposed_pbpp_wosm": Bitswosm,
                "pmsssim": pMSSSIM,
                "mcpsnr": mcPSNR, "warppsnr": warpPSNR, "mvbpp": mvBits,
                "resbpp": resBits, "mcmsssim": mcMSSSIM, "warmsssim": warpMSSSIM,
                "ienc": iEnc, "idec": iDec, "pent": pEnc,
                "pdec": pDec, "ent": Enc, "dec": Dec,
                "smbpp": smBits, "sment": smEnc, "smdec": smDec, "bpp2l": Bits2l,
            }
            output = {
                "name": f'{test_tgt}_{indicator.upper()}_{lambda2}',
                "description": "Inference (ans)",
                "results": results,
            }
            with open(os.path.join(result_save_path, f'{test_tgt}_{indicator.upper()}_{lambda2}.json'),
                      'w', encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)

            porposed_psnr.append(np.average(PSNR))
            porposed_bpp.append(np.average(Bits))
            porposed_pbpp_wosm.append(np.average(Bitswosm))
            porposed_msssim.append(np.average(MSSSIM))
            porposed_ipsnr.append(np.average(iPSNR))
            porposed_ibpp.append(np.average(iBits))
            porposed_imsssim.append(np.average(iMSSSIM))
            porposed_ppsnr.append(np.average(pPSNR))
            porposed_pbpp.append(np.average(pBits))
            porposed_pmsssim.append(np.average(pMSSSIM))

            porposed_mcpsnr.append(np.average(mcPSNR))
            porposed_warppsnr.append(np.average(warpPSNR))
            porposed_mvbpp.append(np.average(mvBits))
            porposed_resbpp.append(np.average(resBits))
            porposed_mcmsssim.append(np.average(mcMSSSIM))
            porposed_warmsssim.append(np.average(warpMSSSIM))
            porposed_ienc.append(np.average(iEnc))
            porposed_idec.append(np.average(iDec))
            porposed_pent.append(np.average(pEnc))
            porposed_pdec.append(np.average(pDec))
            porposed_ent.append(np.average(Enc))
            porposed_dec.append(np.average(Dec))

            porposed_smbpp.append(np.average(smBits))
            porposed_sment.append(np.average(smEnc))
            porposed_smdec.append(np.average(smDec))
            porposed_bpp2l.append(np.average(Bits2l))
    log_txt.close()

    print(porposed_bpp)
    print(porposed_psnr)
    print(porposed_msssim)
    results = {
        "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim,
        "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
        "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "porposed_pbpp_wosm": porposed_pbpp_wosm,
        "pmsssim": porposed_pmsssim,
        "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
        "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
        "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
        "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
        "smbpp": porposed_smbpp, "sment": porposed_sment, "smdec": porposed_smdec, "bpp2l": porposed_bpp2l,
    }
    output = {
        "name": f'{test_tgt}',
        "description": "Inference (ans)",
        "results": results,
    }
    with open(os.path.join(result_save_path, f'{test_tgt}_{indicator.upper()}.json'),
              'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)

    return None


if __name__ == "__main__":
    pass
    # get_result_of_DeepSVC('mse', 'HEVC_D')

    for indicator in ['mse', 'msssim']:
        for tgt in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E', 'UVG', 'MCL-JCV', 'VTL']:
            get_result_of_DeepSVC(indicator, tgt)

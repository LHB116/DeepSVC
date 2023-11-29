import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import utils
import Learner

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main():
    args = utils.get_args()
    # 1. train with key frame coded with bpg  lambda=2048
    Learner.HZHTrainer_1(args).train()

    # 2. train with key frame coded with key frame coded with ICIP2020  PSNR/MS-SSIM
    # Learner.Trainer_ICIP2020ResB_WSM_PSNRv2(args).train()
    # Learner.Trainer_ICIP2020ResB_WSM_MSSSIMv2(args).train()
    return 0


if __name__ == "__main__":
    main()

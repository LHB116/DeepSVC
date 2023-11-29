import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import random


class VimeoDataset(Dataset):
    def __init__(self, root, model_type='PSNR', transform=None, split="train", QP=None, Level=None, mf=5, return_orgi=False):
        assert split == 'train' or 'test'
        assert model_type == 'PSNR' or 'MSSSIM'
        if transform is None:
            raise Exception("Transform must be applied")
        if (model_type == 'PSNR' and QP is None) or (model_type == 'MSSSIM' and Level is None):
            raise Exception("QP or Level must be specified")

        self.max_frames = mf  # for Vimeo DataSet
        self.return_orgi = return_orgi
        self.QP = QP
        self.Level = Level
        self.transform = transform
        self.model_type = model_type

        self.file_name_list = os.path.join(root, f'sep_{split}list.txt')
        self.frames_dir = [os.path.join(root, 'sequences', x.strip())
                           for x in open(self.file_name_list, "r").readlines()]

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index]
        frame_paths = []
        for i in range(1, self.max_frames + 1):
            if i == 1:
                if self.model_type == "PSNR":
                    if self.return_orgi:
                        frame_paths.append(os.path.join(sample_folder, f'im{i}.png'))
                    else:
                        frame_paths.append(
                            os.path.join(sample_folder.replace('sequences', 'bpg'), f'im1_bpg444_QP{self.QP}.png'))
                elif self.model_type == "MSSSIM":
                    frame_paths.append(os.path.join(sample_folder, 'CA_Model', f'im1_level{self.Level}_ssim.png'))
            else:
                frame_paths.append(os.path.join(sample_folder, f'im{i}.png'))

        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        frames = self.transform(frames)
        frames = torch.chunk(frames, chunks=self.max_frames, dim=0)
        return frames

    def __len__(self):
        return len(self.frames_dir)


def get_dataset(args, part='all', mf=5, return_orgi=False, crop=True):
    QP, I_level = 0, 0
    if args.l_PSNR == 256:
        QP = 37
    elif args.l_PSNR == 512:
        QP = 32
    elif args.l_PSNR == 1024:
        QP = 27
    elif args.l_PSNR == 2048:
        QP = 22

    if args.l_MSSSIM == 8:
        I_level = 2
    elif args.l_MSSSIM == 16:
        I_level = 3
    elif args.l_MSSSIM == 32:
        I_level = 5
    elif args.l_MSSSIM == 64:
        I_level = 7
    if crop:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(args.image_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(args.image_size[0]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    training_set = VimeoDataset(root=args.dataset_root,
                                model_type=args.mode_type,
                                transform=train_transforms,
                                split="train",
                                QP=QP,
                                Level=I_level,
                                mf=mf,
                                return_orgi=return_orgi,
                                )
    valid_set = VimeoDataset(root=args.dataset_root,
                             model_type=args.mode_type,
                             transform=test_transforms,
                             split="test",
                             QP=QP,
                             Level=I_level,
                             mf=mf,
                             return_orgi=return_orgi,
                             )
    if part == 'all':
        return training_set, valid_set
    elif part == 'train':
        return training_set
    elif part == 'valid':
        return valid_set


class VIDDataset(Dataset):
    def __init__(self, root, transform=None, split="train", QP=None):
        assert split == 'train' or 'val'
        if transform is None:
            raise Exception("Transform must be applied")

        self.root = root
        self.max_frames = 5  # for VID DataSet
        self.QP = QP
        self.transform = transform

        self.file_name_list = os.path.join(root, 'VID', f'{split}.txt')
        self.frames_dir = [x.strip() for x in open(self.file_name_list, "r").readlines()]

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index].replace('\\', '/')
        print(self.root, sample_folder)
        if sample_folder.split('/')[-1] == '000000.JPEG':
            print(sample_folder.replace('000000.JPEG', 'bpg/000000.JPEG'))
        elif sample_folder.split('/')[-1] == '000004.JPEG':
            print(sample_folder.replace('000004.JPEG', 'bpg/000004.JPEG'))
        # exit()

        start = 0 if sample_folder.split('/')[-1] == '000000.JPEG' else 4
        frame_paths = []
        for i in range(start, start + self.max_frames):
            if i == 0:
                temp = ''
                if sample_folder.split('/')[-1] == '000000.JPEG':
                    temp = sample_folder.replace('000000.JPEG', f'bpg/000000_bpg444_QP{self.QP}.JPEG')
                elif sample_folder.split('/')[-1] == '000004.JPEG':
                    temp = sample_folder.replace('000004.JPEG', f'bpg/000004_bpg444_QP22{self.QP}.JPEG')
                frame_paths.append(os.path.join(self.root, temp))
            else:
                temp = ''
                if sample_folder.split('/')[-1] == '000000.JPEG':
                    temp = sample_folder.replace('000000.JPEG', f'00000{i}.JPEG')
                elif sample_folder.split('/')[-1] == '000004.JPEG':
                    temp = sample_folder.replace('000004.JPEG', f'00000{i}.JPEG')
                frame_paths.append(os.path.join(self.root, temp))

        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        frames = self.transform(frames)
        frames = torch.chunk(frames, chunks=self.max_frames, dim=0)
        return frames

    def __len__(self):
        return len(self.frames_dir)


def get_vid_dataset(args, part='all'):
    QP = 0
    if args.l_PSNR == 256:
        QP = 37
    elif args.l_PSNR == 512:
        QP = 32
    elif args.l_PSNR == 1024:
        QP = 27
    elif args.l_PSNR == 2048:
        QP = 22

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Pad(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    training_set = VIDDataset(root=args.dataset_root,
                              transform=train_transforms,
                              split="train",
                              QP=QP,
                              )
    valid_set = VIDDataset(root=args.dataset_root,
                           transform=test_transforms,
                           split="val",
                           QP=QP,
                           )
    if part == 'all':
        return training_set, valid_set
    elif part == 'train':
        return training_set
    elif part == 'valid':
        return valid_set


class VimeoDataset1(Dataset):
    def __init__(self, root, model_type='PSNR', transform=None, split="train", QP=None, Level=None, return_orgi=False,
                 mf=5):
        assert split == 'train' or 'test'
        assert model_type == 'PSNR' or 'MSSSIM'
        if transform is None:
            raise Exception("Transform must be applied")
        if (model_type == 'PSNR' and QP is None) or (model_type == 'MSSSIM' and Level is None):
            raise Exception("QP or Level must be specified")

        self.max_frames = mf  # for Vimeo DataSet
        self.QP = QP
        self.Level = Level
        self.transform = transform
        self.model_type = model_type
        self.return_orgi = return_orgi

        self.file_name_list = os.path.join(root, f'sep_{split}list.txt')
        self.frames_dir = [os.path.join(root, 'sequences', x.strip())
                           for x in open(self.file_name_list, "r").readlines()]
        # print(self.frames_dir[:4])

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index]
        frame_paths = []
        for i in range(self.max_frames):
            if i == 0:
                if self.model_type == "PSNR":
                    if self.return_orgi:
                        frame_paths.append(os.path.join(sample_folder, f'im{i + 1}.png'))
                        # frame_paths.append(os.path.join(sample_folder, 'bpg', f'im1_bpg444_QP{self.QP}.png'))
                        frame_paths.append(
                            os.path.join(sample_folder.replace('vimeo_septuplet/sequences', 'ICIP2020_i_mse'), f'im1_1.png'))
                        # print(frame_paths)
                    else:
                        frame_paths.append(os.path.join(sample_folder, 'bpg', f'im1_bpg444_QP{self.QP}.png'))
                elif self.model_type == "MSSSIM":
                    frame_paths.append(os.path.join(sample_folder, 'CA_Model', f'im1_level{self.Level}_ssim.png'))
            else:
                frame_paths.append(os.path.join(sample_folder, f'im{i + 1}.png'))
        # print(frame_paths)
        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )

        frames = self.transform(frames)
        if self.return_orgi:
            frames = torch.chunk(frames, chunks=self.max_frames + 1, dim=0)
        else:
            frames = torch.chunk(frames, chunks=self.max_frames, dim=0)
        return frames

    def __len__(self):
        return len(self.frames_dir)


def get_dataset1(args, part='all', return_orgi=True, mf=7):
    QP, I_level = 0, 0
    if args.l_PSNR == 256:
        QP = 37
    elif args.l_PSNR == 512:
        QP = 32
    elif args.l_PSNR == 1024:
        QP = 27
    elif args.l_PSNR == 2048:
        QP = 22

    if args.l_MSSSIM == 8:
        I_level = 2
    elif args.l_MSSSIM == 16:
        I_level = 3
    elif args.l_MSSSIM == 32:
        I_level = 5
    elif args.l_MSSSIM == 64:
        I_level = 7

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Pad(),
            transforms.RandomCrop(args.image_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(args.image_size[0]),
        ]
    )

    training_set = VimeoDataset1(root=args.dataset_root,
                                 model_type=args.mode_type,
                                 transform=train_transforms,
                                 split="train",
                                 QP=QP,
                                 Level=I_level,
                                 return_orgi=return_orgi,
                                 mf=mf
                                 )
    valid_set = VimeoDataset1(root=args.dataset_root,
                              model_type=args.mode_type,
                              transform=test_transforms,
                              split="test",
                              QP=QP,
                              Level=I_level,
                              return_orgi=return_orgi,
                              mf=mf
                              )
    if part == 'all':
        return training_set, valid_set
    elif part == 'train':
        return training_set
    elif part == 'valid':
        return valid_set


if __name__ == "__main__":
    pass
    size = 256

    # vid_path = 'D:/DataSet/VIDTrainCodec'
    transforms1 = transforms.Compose([transforms.ToTensor()])
    # dataset = VIDDataset(vid_path, transform=transforms1, split="train", QP=22)
    # print(len(dataset[0]), dataset[0][0].shape)

    dataset = VimeoDataset1('/home/user/桌面/LHB/vimeo_septuplet',
                           model_type='PSNR',
                           transform=transforms1,
                           split="train",
                           QP=22,
                            return_orgi=True,
                           )
    print(len(dataset[0]), dataset[0][0].shape)

# DeepSVC: Deep Scalable Video Coding for Both Machine and Human Vision

If our open source codes are helpful for your research, please cite our [paper](https://dl.acm.org/doi/10.1145/3581783.3612500):

```
@inproceedings{lin2023deepsvc,
  title={DeepSVC: Deep Scalable Video Coding for Both Machine and Human Vision},
  author={Lin, Hongbin and Chen, Bolin and Zhang, Zhichen and Lin, Jielian and Wang, Xu and Zhao, Tiesong},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={9205--9214},
  year={2023}
}
```

## Dependency

- see env.txt


## Test codes

### Semantic Layer

- The code should run with [mmtracking](https://github.com/open-mmlab/mmtracking), copy the codes in temporal_roi_align.py to [selsa.py](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/models/vid/selsa.py). Then run [test.py](https://github.com/open-mmlab/mmtracking/blob/master/tools/test.py) with config file [selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py) . Please see instructions in [docs](https://github.com/open-mmlab/mmtracking/tree/master/docs).

### Structure and Texture Layer

- Run test_video.py, please change data path in the file.

## Training your own models

### Semantic Layer

- The code should run with [mmtracking](https://github.com/open-mmlab/mmtracking), copy the codes in temporal_roi_align.py to [selsa.py](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/models/vid/selsa.py). Then run [train.py ](https://github.com/open-mmlab/mmtracking/blob/master/tools/train.py) with config file [selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py) . Please see instructions in [docs](https://github.com/open-mmlab/mmtracking/tree/master/docs).

### Structure and Texture Layer,training the PSNR/MS-SSIM models

- Download the training data. We train the models on the [Vimeo90k dataset](https://github.com/anchen1011/toflow) ([Download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)). 

- Run main.py to train the PSNR/MS-SSIM models. We first pretrian model with key frame coded with bpg and lambda=2048. Then load the pretrianed weights, train with key frame coded with key frame coded with AI codecs (in image_model.py). More detail see main.py.

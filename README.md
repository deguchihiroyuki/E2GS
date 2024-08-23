# Event Enhanced Gaussian Splatting(E2GS)
Hiroyuki Deguchi*, Mana Masuda*, Takuya Nakabayshi, Hideo Saito (* indicates equal contribution)<br>
## Requirements and Setup
Our code is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). Please refer to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) Requirements and Setup.
## Data
We use [E2NeRF](https://github.com/iCVTEAM/E2NeRF) synthetic dataset, so first dowload it. 
After that, use [COLMAP](https://github.com/colmap/colmap) to make the point cloud. As for the synthetic data, we have to generate only a point cloud, so it works even if we input blury image directly to COLMAP. However, if you prefer to use deblurred images, it would be good to refer to [EDI](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) model.
Data Structere is like below.
```
|--data
  |--chair
    |--images
      |--r_0.png
      |--r_2.png
      |--...
    |--sparse
      |--0
        |--points3D.bin
    |--events.pt
    |--transform_test.json
    |--transform_train.json
  |--ficus
  |--...
  
```
As for the realworld data, we have to deblur the blured image using [EDI](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) model and estimate 5 camera poses in exposure time per image.
## Citation  
```
@inproceedings{E2GS,
title={E2GS:Event Enhanced Gaussian Splatting},
author={Hiroyuki Deguchi and Mana Masuda and Takuya Nakabayashi and Hideo Saito},
booktitle={IEEE International Conference on Image Processing},
year={2024}
}
```

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

The data I have processed can be downloaded from the following link.

[Download the input data](https://keio.box.com/s/8mvtjcfj2ekc0z0d9l71xcf20thdl143)

(images_d in each directory is a directory which contains images deblurred by [EDI](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) model.)
### Running

To run the optimizer, simply use

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python render.py -m <path to trained model> # Generate input view renderings
python render_novel.py -m <path to trained model> # Generate novel view renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
## Citation  
```
@inproceedings{E2GS,
title={E2GS:Event Enhanced Gaussian Splatting},
author={Hiroyuki Deguchi and Mana Masuda and Takuya Nakabayashi and Hideo Saito},
booktitle={IEEE International Conference on Image Processing},
year={2024}
}
```

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/generalized-zero-shot-learning-on-awa2)](https://paperswithcode.com/sota/generalized-zero-shot-learning-on-awa2?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/zero-shot-learning-on-awa2)](https://paperswithcode.com/sota/zero-shot-learning-on-awa2?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/generalized-zero-shot-learning-on-cub-200)](https://paperswithcode.com/sota/generalized-zero-shot-learning-on-cub-200?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/zero-shot-learning-on-cub-200-2011)](https://paperswithcode.com/sota/zero-shot-learning-on-cub-200-2011?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/generalized-zero-shot-learning-on-oxford-102-1)](https://paperswithcode.com/sota/generalized-zero-shot-learning-on-oxford-102-1?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/zero-shot-learning-on-oxford-102-flower)](https://paperswithcode.com/sota/zero-shot-learning-on-oxford-102-flower?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/generalized-zero-shot-learning-on-sun)](https://paperswithcode.com/sota/generalized-zero-shot-learning-on-sun?p=latent-embedding-feedback-and-discriminative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-embedding-feedback-and-discriminative/zero-shot-learning-on-sun-attribute)](https://paperswithcode.com/sota/zero-shot-learning-on-sun-attribute?p=latent-embedding-feedback-and-discriminative)




# New Paths for Zero-Shot Learning Research

#### [Nicola Carletti], [Iacopo Simoncini]####


Zero-shot learning (ZSL) is a problem setup in machine learning which aims to train a model for classifying data samples under the condition that some
  output classes are unknown during supervised learning. To reach this goal, ZSL leverages semantic information of
  both seen (source) and unseen (target) classes to bridge the gap between both seen and unseen classes. Since its introduction, many
  ZSL models have been formulated. In this paper, we have analyzed the best method at the state of the art to understand if 
  current dataset in use should be improved to reach better accuracy in recognizing seen and unseen classes.



## Prerequisites
+ Python 3.6
+ Pytorch 0.3.1
+ torchvision 0.2.0
+ h5py 2.10
+ scikit-learn 0.22.1
+ scipy=1.4.1
+ numpy 1.18.1
+ numpy-base 1.18.1
+ pillow 5.1.0

## Installation
The model is built in PyTorch 0.3.1 and tested on Ubuntu 16.04 environment (Python3.6, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n tfvaegan python=3.6
conda activate tfvaegan
pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision==0.2.0 scikit-learn==0.22.1 scipy==1.4.1 h5py==2.10 numpy==1.18.1
```

## Data preparation



Download CUB, AWA, FLO and SUN features from the drive link shared below.
```
link: https://drive.google.com/drive/folders/16Xk1eFSWjQTtuQivTogMmvL3P6F_084u?usp=sharing

Save the datasets on your drive

Run Split_generator.ipynb in the notebook folder to obtain the unseen class array of the split you want to use. Then paste this array in the image_util.py file, at the position we have specified with a comment.

For the attribute splits you have to follow the instruction that we put on a comment on image_util.py


## Scripts
Use the scripts in the scripts folder to run the code.

Remember to select the correct script for each split, it changes for the attributes split!

# New Paths for Zero-Shot Learning Research

#### Nicola Carletti, Iacopo Simoncini 


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
Use this scripts to run the code.

###Datasets split:

###AWA: python train_images.py --gammaD 10 --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataroot datasets --dataset AWA2 --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01

###CUB: python train_images.py --gammaD 10 --gammaG 10 --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot datasets --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2

###SUN: python train_images.py --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot datasets --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001

###Attributes deleting

###AWA: python train_images.py --gammaD 10 --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataroot datasets --dataset AWA2 --batch_size 64 --nz 80 --latent_size 80 --attSize 80 --resSize 2048 --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01

###CUB: python train_images.py --gammaD 10 --gammaG 10 --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot datasets --dataset CUB --nclass_all 200 --batch_size 64 --nz 307 --latent_size 307 --attSize 307 --resSize 2048 --syn_num 300 --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2

###SUN: python train_images.py --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 97 --latent_size 97 --attSize 97 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot datasets --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001

###Attributes Merging

###AWA: python train_images.py --gammaD 10 --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataroot datasets --dataset AWA2 --batch_size 64 --nz 81 --latent_size 81 --attSize 81 --resSize 2048 --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01

###CUB: python train_images.py --gammaD 10 --gammaG 10 --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot datasets --dataset CUB --nclass_all 200 --batch_size 64 --nz 308 --latent_size 308 --attSize 308 --resSize 2048 --syn_num 300 --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2

###SUN: python train_images.py --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 98 --latent_size 98 --attSize 98 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot datasets --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001

Remember to select the correct script for each split, it changes for the attributes split!


For further information contact us!!!

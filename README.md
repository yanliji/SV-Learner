# SV-Learner: Support-Vector Contrastive Learning for Robust Learning with Noisy Labels

<img src="https://github.com/yanliji/SV-Learner/blob/main/figure/framework.jpg"  width="800px" />

This is the official PyTorch implementation of 
Transactions on Knowledge and Data Engineering paper (SV-Learner: Support-Vector Contrastive Learning for Robust Learning with Noisy Labels).

## Abstract
Noisy-label data inevitably gives rise to confusion in various perception applications. In this work, we revisit the theory of support vector machines (SVM) which mines support vectors to build the maximum-margin hyperplane for robust classification, and propose a robust-to-noise deep learning framework, SV-Learner, including the Support Vector Contrastive Learning (SVCL) and Support Vector-based Noise Screening (SVNS). The SV-Learner mines support vectors to solve the learning problem with noisy labels (LNL) reliably. Support Vector Contrastive Learning (SVCL) adopts support vectors as positive and negative samples, driving robust contrastive learning to enlarge the feature distribution margin for learning convergent feature distributions. Support Vector-based Noise Screening (SVNS) uses support vectors with valid labels to assist in screening noisy ones from confusable samples for reliable clean-noisy sample screening. Finally, Semi-Supervised classification is performed to realize the recognition of clean noisy samples. Extensive experiments are evaluated on CIFAR-10, CIFAR-100, Clothing1M, and Webvision datasets, demonstrating our proposed approach's effectiveness.

## Preparation
- numpy
- opencv-python
- Pillow
- torch
- torchnet
- sklearn

Our code is written by Python, based on Pytorch (Version ≥ 1.6).

## Datasets

For CIFAR 10 / 100 datasets, one can directly run the shell codes.

For Clothing1M and Webvision, you need to download them from their corresponsing website.

## Usage

### Training for CIFAR-10/100
Example runs on CIFAR-10 dataset with 20% symmetric noise:
```
  python Train_cifar_svmfix_svm.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.2 --lambda_u=0
```

Example runs on CIFAR-10 dataset with 50% symmetric noise:
```
  python Train_cifar_svmfix_svm.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 --lambda_u=25
```

Example runs on CIFAR-100 dataset with 90% symmetric noise:
```
  python Train_cifar_svmfix_svm.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 --lambda_u=150
```

Example runs on CIFAR-10 dataset with 40% asymmetric noise:
```
  python Train_cifar_svmfix_svm.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'asym' --r 0.4 --lambda_u=25
```

## Results
<img src="https://github.com/yanliji/SV-Learner/blob/main/figure/result_cifar10.jpg"  width="400px" /> <img src="https://github.com/yanliji/SV-Learner/blob/main/figure/result_cifar100.jpg"  width="360px" />

## Visualization of embedded features
Visualization of embedded features on CIFAR-10 with 90% symmetric noise.

90% symmetric noise.

<img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/tsne-90%25.png"  width="360px" />

## Visualization of support vectors

The t-SNE visualization of support vectors in CIFAR-10 with 50% symmetric noise.

<img src="https://github.com/yanliji/SV-Learner/blob/main/figure/support%20vectors.jpg"  width="360px" />

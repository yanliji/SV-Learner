# SV-Learner: Support Vector-drived Contrastive Learning for Robust Learning with Noisy labels

<img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/SV-Learner.png"  width="800px" />

This is the official PyTorch implementation of IJCAI 2023 paper (SV-Learner: Support Vector-drived Contrastive Learning for Robust Learning with Noisy labels).

## Abstract
Noisy-label data inevitably gives rise to confusion in various perception applications. In this paper, we propose a robust-to-noise framework SV-Learner to solve the problem of recognition with noisy labels. In particular, we first design a Dynamic Noisy Sample Selection (DNSS) solution for learning more robust classification boundaries, which dynamically determines the filter rates of classifiers for reliable noisy sample selection based on curriculum learning. Inspired by support vector machines (SVM), we propose a Support Vector driven Contrastive Learning (SVCL) approach that mines support vectors near classification boundaries as negative samples to drive contrastive learning. These support vectors expand the margin between different classes for contrastive learning, therefore better promoting the robust detection of noise samples. Finally, a Dynamic Semi-Supervised Classification (DSSC) module is presented to realize noisy-label recognition. In comparison with the state-of-the-art approaches, the proposed SV-Learner achieves the best performance in multiple datasets, including the CIFAR-10, CIFAR-100, Clothing1M, and Webvision datasets. Extensive experiments demonstrate the effectiveness of our proposed method. 

## Preparation
- numpy
- opencv-python
- Pillow
- torch
- torchnet
- sklearn

Our code is written by Python, based on Pytorch (Version ≥ 1.6).

## Datasets

For CIFAR datasets, one can directly run the shell codes.

For Clothing1M and Webvision, you need to download them from their corresponsing website.

## Usage

### Training for CIFAR-10/100
Example runs on CIFAR-10 dataset with 20% symmetric noise:
```
  python Train_cifar_sv-learner.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 --lambda_u=0
```

Example runs on CIFAR-100 dataset with 90% symmetric noise:
```
  python Train_cifar_sv-learner.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 --lambda_u=150
```

Example runs on CIFAR-10 dataset with 40% asymmetric noise:
```
  python Train_cifar_sv-learner.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'asym' --r 0.4 --lambda_u=25
```
### Training for Clothing1M
The performance improvement for clothing1M sometimes depends on the length of warmup and the initialization of SVCL, as longer period of standard CE-based training can lead to memorization. We didn't carefully tune the best number of epochs. A smaller number of training epochs (e.g. 120~200 epochs with a slightly larger learning rate) can also produce good results, actually.
```
python Train_clothing1M_sv-learner.py --batch_size 64 --num_epochs 200    --lambda_u=0
```

### Training for Webvision
```
python Train_webvision_sv-learner.py.py  --batch_size 64 --num_epochs 100    --lambda_u=0
```

## Results
<img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/result_cifar10.jpg"  width="400px" /> <img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/result_cifar100.jpg"  width="360px" />

## Visualization
Visualization of embedded features on CIFAR-10 with 20%-90% symmetric noise.

20% symmetric noise.

<img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/tsne-20%25.png"  width="360px" /> 

90% symmetric noise.

<img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/tsne-90%25.png"  width="360px" />


## License
This project is licensed under the terms of the MIT license.

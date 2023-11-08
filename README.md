# SV-Learner: Support-Vector Contrastive Learning for Robust Learning with Noisy Labels

<img src="https://github.com/yanliji/SV-Learner/blob/main/figures/framework.pdf"  width="800px" />

This is the official PyTorch implementation of IJCAI 2023 paper (SV-Learner: Support Vector-drived Contrastive Learning for Robust Learning with Noisy labels).

## Abstract
Noisy-label data inevitably gives rise to confusion in various perception applications. In this work, we revisit the theory of support vector machines (SVM) which mines support vectors to build the maximum-margin hyperplane for robust classification, and propose a robust-to-noise deep learning framework, SV-Learner, including the Support Vector Contrastive Learning (SVCL) and Support Vector-based Noise Screening (SVNS). The SV-Learner mines support vectors to solve the learning problem with noisy labels (LNL) reliably. Support Vector Contrastive Learning (SVCL) adopts support vectors as positive and negative samples, driving robust contrastive learning to enlarge the feature distribution margin for learning convergent feature distributions. Support Vector-based Noise Screening (SVNS) uses support vectors with valid labels to assist in screening noisy ones from confusable samples for reliable clean-noisy sample screening. Finally, Semi-Supervised classification is performed to realize the recognition of clean noisy samples. Extensive experiments are evaluated on CIFAR- 10, CIFAR-100, Clothing1M, and Webvision datasets, and they demonstrate the effectiveness of our proposed approach.

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

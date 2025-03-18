#A Few-Shot High-Resolution Remote Sensing Image Semantic Segmentation Method


## Results
### Erhai 
To evaluate the effectiveness of the proposed model optimization, we conducted experiments on the Erhai remote sensing image dataset using the improved HRNet-based backbone model (D_HRNetV2_PSA). Under the same experimental settings, the model was trained using the entire labeled training set of the Erhai dataset (8,800 images) to ensure consistency and reliability of the method. To maintain a fair comparison, the same parameter configurations, data augmentation strategies, and evaluation datasets were used across all models. Different backbone architectures were compared to assess their effectiveness, with GFLOPs used to measure computational complexity and MIoU used to evaluate segmentation performance. The experimental results are presented in Table 1.
Table 1. Performance of DeepLabV3+ with Different Backbone Networks on the Erhai Test Set
| Weights     | Backbone             | GFLOPs | MIoU  |
|-------------|----------------------|--------|-------|
| DeepLabv3+  | ImageNet ResNet-101   | 1661.6 | 79.04 |
| DeepLabv3+  | ImageNet Resnet101_ibn_a | 1778.7 | 79.83 |
| DeepLabv3+  | ImageNet ResNext-101  | 1788.2 | 80.01 |
| DeepLabv3+  | ImageNet Xception71   | 1344.6 | 80.22 |
| DeepLabv3+  | ImageNet HRNetV2-W48  | 1160.1 | 81.31 |
| DeepLabv3+  | ImageNet HRNetV2-W48_PSA | 1262.1 | 81.93 |

### Experimental Results of Semi-Supervised Learning and Multi-Stage Knowledge Distillation
To validate the effectiveness of multi-stage knowledge distillation and semi-supervised learning strategies, this section presents experiments conducted under the same experimental settings using 10% of the labeled training subset of the Erhai dataset (880 images). To ensure the consistency and reliability of the method, the experiments maintain the same parameter configurations, data augmentation strategies, and evaluation dataset. This section focuses on analyzing the performance differences with and without semi-supervised learning under different knowledge distillation strategies, comparing the IoU per category and overall MIoU. The experimental results are presented in Table 2.
| Knowledge Distillation Strategy/Semi-Supervised Learning | Background | Road   | Farmland | Building | Water  | Grassland | MIoU  |
|----------------------------------------------------------|------------|--------|----------|----------|--------|-----------|-------|
| No Distillation/No                                       | 71.20      | 39.17  | 60.69    | 70.40    | 70.81  | 54.10     | 61.07 |
| No Distillation/Yes                                      | 74.11      | 56.37  | 78.90    | 79.25    | 77.53  | 65.88     | 72.03 |
| ImageNet/No                                              | 75.27      | 51.94  | 70.63    | 76.09    | 73.91  | 62.89     | 68.46 |
| ImageNet/Yes                                             | 74.82      | 60.09  | 81.08    | 81.34    | 78.96  | 67.68     | 73.99 |
| HW17/No                                                  | 74.29      | 55.29  | 74.75    | 78.26    | 75.15  | 65.01     | 70.46 |
| HW17/Yes                                                 | 76.24      | 62.17  | 80.67    | 83.36    | 78.04  | 67.29     | 74.63 |
| ImageNet_HW17_EWC/No                                      | 75.30      | 60.34  | 80.61    | 81.57    | 78.51  | 67.93     | 74.04 |
| ImageNet_HW17_EWC/Yes                                     | 76.40      | 65.30  | 84.82    | 83.51    | 81.41  | 70.87     | 77.05 |





### Cityscapes

To further validate the effectiveness of the proposed semi-supervised learning and multi-stage knowledge distillation model, we conducted comparative experiments on the public Cityscapes dataset and compared the MIoU of existing methods with our approach (Table 3). All methods were implemented based on the DeepLabV3+ model architecture, utilizing sliding window evaluation and the Online Hard Example Mining (OHEM) loss function. The dataset partitioning followed the standard protocol described in[33], where the proportions of labeled training samples were set to 1/16, 1/8, 1/4, and 1/2 of the full dataset.

Table 3. Comparative Experiments with Existing Methods on the Cityscapes Dataset
| Method                 | Pre-trained weights    | Backbone     | 1/16  | 1/8   | 1/4   | 1/2   |
|------------------------|------------------------|--------------|-------|-------|-------|-------|
| SupBaseline            | ImageNet               | ResNet-101   | 66.3  | 72.8  | 75.0  | 78.0  |
| MT[34]                 | ImageNet               | ResNet-101   | 68.08 | 73.71 | 76.53 | 78.59 |
| CCT[35]                | ImageNet               | ResNet-101   | 69.64 | 74.48 | 76.35 | 78.29 |
| GCT[36]                | ImageNet               | ResNet-101   | 66.90 | 72.96 | 76.45 | 78.58 |
| U2PL[37]               | ImageNet               | ResNet-101   | 74.9  | 76.5  | 78.5  | 79.1  |
| UniMatch[33]           | ImageNet               | ResNet-101   | 76.6  | 77.9  | 79.2  | 79.5  |
| D_HRNetV2_PSA(Ours)    | Imagenet_GTAV_EWC      | HRNetV2-W48  | 78.1  | 79.3  | 79.4  | 80.3  |



## Getting Started

### Installation

```bash

conda create -n AFSH python=3.10.4
conda activate AFSH
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

[ResNet-101](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing) | [Xception-65](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi)

├── ./pretrained
    ├── resnet101.pth
    ├── resnet101_ibn_a.pth
    ├── resnext101.pth
    ├── xception71.pth
    ├── hrnetv2_w48.pth
    └── hrnetv2_w48_psa.pth




**The groundtruth masks have already been pre-processed by us. You can use them directly.**

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
    
├── [Your COCO Path]
    ├── train2017
    ├── val2017
    └── masks
```

## Usage

### UniMatch

```bash

#Training
# python teacher_emc.py
# python  sem_teacher.py


#Test 
#  python test.py


```

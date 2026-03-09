# Transfer Learning for Multi-Label Medical Image Classification

## Overview

This repository contains the code and findings for a deep learning project exploring transfer learning techniques for multi-label medical image classification. The goal was to classify retinal images from the ODIR dataset into three disease categories: diabetic retinopathy (DR), glaucoma, and age-related macular degeneration (AMD). The project specifically tackles challenges related to scarce and imbalanced medical data by experimenting with different architectures, fine-tuning strategies, attention mechanisms, and loss functions.

See [report.pdf](report.pdf) for a detailed write-up of the project.

## Main findings

Based on the experimental evaluations, the project yielded the following insights:

- **Fine-tuning strategies**: Full fine-tuning of the entire network significantly outperformed tuning only the classifier head. This indicates that the domain shift from general object recognition to medical imaging requires retraining deep semantic features.

- **Handling class imbalance**: Utilizing specialized loss functions improved overall performance compared to standard BCE loss. While Class-Balanced Loss achieved high validation performance, Focal Loss demonstrated better generalization on unseen test distributions.

- **Attention mechanisms**: Integrating Squeeze-and-Excitation (SE) blocks provided lightweight, channel-wise feature recalibration that significantly outperformed baseline models. In contrast, Multi-Head Attention (MHA) was less effective, likely due to its higher parameter count causing overfitting on the small dataset.

- **Architectures and ensembling**: The Swin Transformer proved to be highly effective despite the limited data, whereas smaller models like MobileNetV3 lacked sufficient capacity. The best overall performance was achieved through an ensemble model that combined the complementary strengths and distinct feature representations of the Swin Transformer and the ResNet-18 model with SE attention. Additionally, while the higly optimized Efficientnet initially outperformed ResNet-18, it did not respond well to fine-tuning, which highlights the importance of selecting architectures that are amenable to transfer learning and adaptation to the target domain.

## Repository structure

The codebase is structured into modular PyTorch scripts for training, evaluating, and ensembling models:

* **`src/dataset.py`**: Contains the `RetinaMultiLabelDataset` class, which handles loading the images, applying transformations (data augmentation and normalization), and formatting the multi-label targets into PyTorch tensors.
* **`src/models.py`**: Defines the model builder. It supports multiple backbones (`resnet18`, `efficientnet_b0`, `mobilenet_v3_small`, and `swin_t`) and implements custom attention pooling layers, including Squeeze-and-Excitation (SE) and Multi-Head Attention (MHA) blocks.
* **`train.py`**: The primary training script. Handles data loading, model initialization, and the training loop. It supports hyperparameter tuning, different fine-tuning strategies (freezing the backbone vs. full fine-tuning), and specialized loss functions (BCE, Focal Loss, Class-Balanced Loss).
* **`evaluate.py`**: An inference script to load trained model checkpoints, process a test dataset, and output binary predictions into a CSV file.
* **`ensemble_onsite.py` & `ensemble_offsite.py**`: Scripts that load multiple trained models (specifically Swin Transformer and ResNet-18 with SE Attention) and generate final predictions using a soft-voting ensemble approach.
* **`src/utils.py`**: Contains a `seed_everything` function to ensure reproducibility across Python, NumPy, and PyTorch environments.

## How to run

The effect of fine-tuning:

```python
python train.py --backbone efficientnet --fine_tuning none
python train.py --backbone efficientnet --fine_tuning classifier
python train.py --backbone efficientnet --fine_tuning full
```

The effect of different loss functions:

```python
python train.py --backbone resnet --loss focal
python train.py --backbone resnet --loss balanced
```

Attention mechanisms:

```python
python train.py --backbone resnet --loss balanced --attention se
python train.py --backbone resnet --loss balanced --attention mha
```

Other approaches:

```python
python train.py --backbone swin
python train.py --backbone mobilenet
python ensemble_offsite.py # requires kagone_task4-swin.pt and kagone_task3-1.pt to be in checkpoints/
python ensemble_onsite.py # requires kagone_task4-swin.pt and kagone_task3-1.pt to be in checkpoints/
```
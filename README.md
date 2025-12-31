# Generalizable Semantic Segmentation

A production-grade framework for training and deploying generalizable semantic segmentation models. This project focuses on building robust segmentation models that perform effectively across diverse datasets and real-world scenarios.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Semantic segmentation is a computer vision task that assigns a class label to each pixel in an image. This repository provides a comprehensive framework for:

- Training segmentation models on various datasets
- Evaluating model performance across different domains
- Deploying models to production environments
- Supporting multiple network architectures and backbones
- Handling domain adaptation and generalization challenges

The framework is designed with modularity and scalability in mind, enabling easy extension for custom datasets and architectures.

## Features

### Core Capabilities

- **Multi-Architecture Support**: Support for various encoder-decoder architectures
  - FCN (Fully Convolutional Networks)
  - U-Net
  - DeepLab (v2, v3, v3+)
  - SegNet
  - PSPNet
  
- **Multiple Backbone Networks**:
  - ResNet (18, 34, 50, 101, 152)
  - VGG (16, 19)
  - EfficientNet
  - DenseNet
  
- **Dataset Support**:
  - Cityscapes
  - Pascal VOC
  - ADE20K
  - Camvid
  - Custom datasets with easy integration

- **Advanced Training Features**:
  - Mixed precision training (FP16)
  - Distributed training (multi-GPU, multi-node)
  - Learning rate scheduling
  - Augmentation strategies
  - Checkpoint management and resuming

- **Evaluation Metrics**:
  - Pixel Accuracy
  - Mean Intersection over Union (mIoU)
  - Class-wise IoU
  - Frequency Weighted IoU
  - Confusion matrices

- **Production Features**:
  - Model quantization and pruning
  - ONNX export
  - TensorRT optimization
  - REST API for inference
  - Batch processing capabilities

## Architecture

### System Design

```
┌─────────────────────────────────────────────────┐
│         Semantic Segmentation Framework         │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌────────────────┐       ┌────────────────┐   │
│  │    Data Layer  │       │  Model Layer   │   │
│  ├────────────────┤       ├────────────────┤   │
│  │ • Loaders      │       │ • Encoders     │   │
│  │ • Augmentation │       │ • Decoders     │   │
│  │ • Validation   │       │ • Heads        │   │
│  └────────────────┘       └────────────────┘   │
│           │                      │              │
│           └──────────────────────┘              │
│                      │                          │
│  ┌────────────────────────────────────────┐    │
│  │      Training & Optimization Layer     │    │
│  ├────────────────────────────────────────┤    │
│  │ • Loss Functions                       │    │
│  │ • Optimizers                           │    │
│  │ • Schedulers                           │    │
│  │ • Metrics & Monitoring                 │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│  ┌────────────────────────────────────────┐    │
│  │     Inference & Deployment Layer       │    │
│  ├────────────────────────────────────────┤    │
│  │ • Model Export                         │    │
│  │ • Inference Engine                     │    │
│  │ • API Server                           │    │
│  │ • Batch Processing                     │    │
│  └────────────────────────────────────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Key Components

#### 1. **Data Processing Pipeline**
- Handles multiple dataset formats
- Implements efficient data loading with DataLoader
- Supports online and offline augmentation
- Automatic train/val/test split management

#### 2. **Model Architecture**
- Modular encoder-decoder design
- Pluggable backbone networks
- Customizable decoder modules
- Support for skip connections and feature fusion

#### 3. **Training Engine**
- Distributed training support
- Learning rate scheduling
- Early stopping and checkpointing
- Comprehensive logging and visualization

#### 4. **Evaluation Framework**
- Pixel-level accuracy metrics
- Semantic consistency metrics
- Domain-specific evaluation
- Per-class performance analysis

#### 5. **Inference Pipeline**
- Optimized forward passes
- Batch processing support
- Memory-efficient predictions
- Real-time inference capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- pip or conda package manager

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation.git
cd Generelizable-semantic-segmentation
```

#### 2. Create Virtual Environment

Using Python venv:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using conda:
```bash
conda create -n seg-env python=3.9
conda activate seg-env
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- torch>=1.10.0
- torchvision>=0.11.0
- numpy>=1.19.0
- opencv-python>=4.5.0
- tensorboard>=2.8.0
- tqdm>=4.60.0
- pyyaml>=5.4.0
- scikit-learn>=0.24.0
- Pillow>=8.0.0

#### 4. Install Optional Dependencies

For advanced features:

```bash
# For distributed training
pip install torch-distributed-utils

# For model optimization
pip install onnx onnx-simplifier tensorrt

# For API deployment
pip install flask fastapi uvicorn

# For visualization
pip install matplotlib seaborn tensorboard
```

#### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision {torchvision.__version__}')"
```

## Quick Start

### 1. Prepare Dataset

```bash
# Download dataset (example: Cityscapes)
bash scripts/download_cityscapes.sh /path/to/datasets

# For custom dataset, organize as:
# dataset_root/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
```

### 2. Configure Training

Create a configuration file `configs/my_config.yaml`:

```yaml
model:
  name: "deeplabv3+"
  backbone: "resnet50"
  num_classes: 19
  pretrained: true

dataset:
  name: "cityscapes"
  root: "/path/to/datasets/cityscapes"
  split: "train"
  batch_size: 16
  num_workers: 4

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  loss: "cross_entropy"
  device: "cuda"
  
augmentation:
  enabled: true
  random_flip: true
  random_crop: true
  color_jitter: true
```

### 3. Train Model

```bash
python train.py --config configs/my_config.yaml
```

### 4. Evaluate Model

```bash
python evaluate.py --model_path checkpoints/best_model.pth --dataset_root /path/to/dataset
```

### 5. Run Inference

```bash
python inference.py --model_path checkpoints/best_model.pth --image_path /path/to/image.jpg
```

## Usage Guide

### Training a Model

#### Basic Training

```bash
python train.py \
  --config configs/my_config.yaml \
  --output_dir results/my_experiment
```

#### Resume Training from Checkpoint

```bash
python train.py \
  --config configs/my_config.yaml \
  --checkpoint checkpoints/model_epoch_50.pth \
  --output_dir results/my_experiment
```

#### Distributed Training (Multi-GPU)

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train.py \
  --config configs/my_config.yaml \
  --distributed
```

### Evaluation and Metrics

#### Evaluate on Test Set

```bash
python evaluate.py \
  --model_path checkpoints/best_model.pth \
  --config configs/my_config.yaml \
  --split test
```

#### Generate Segmentation Maps

```bash
python visualize.py \
  --model_path checkpoints/best_model.pth \
  --dataset_root /path/to/dataset \
  --output_dir visualizations/
```

### Inference on New Images

#### Single Image

```bash
python inference.py \
  --model_path checkpoints/best_model.pth \
  --image_path /path/to/image.jpg \
  --output_path results/segmentation.png
```

#### Batch Processing

```bash
python inference.py \
  --model_path checkpoints/best_model.pth \
  --image_dir /path/to/images/ \
  --output_dir results/segmentations/ \
  --batch_size 32
```

#### Real-time Video Processing

```bash
python inference_video.py \
  --model_path checkpoints/best_model.pth \
  --video_path /path/to/video.mp4 \
  --output_path results/segmented_video.mp4
```

### Model Export

#### Export to ONNX

```bash
python export_onnx.py \
  --model_path checkpoints/best_model.pth \
  --output_path models/segmentation.onnx \
  --opset_version 12
```

#### Quantization

```bash
python quantize_model.py \
  --model_path checkpoints/best_model.pth \
  --output_path models/segmentation_quantized.pth \
  --quantization_type int8
```

## Configuration

### Configuration File Structure

```yaml
# Model Configuration
model:
  name: str                    # Model architecture name
  backbone: str               # Backbone network
  num_classes: int            # Number of segmentation classes
  pretrained: bool            # Use pretrained weights
  freeze_backbone: bool       # Freeze backbone during training

# Dataset Configuration
dataset:
  name: str                   # Dataset name
  root: str                   # Dataset root directory
  split: str                  # Train/val/test split
  batch_size: int             # Batch size
  num_workers: int            # DataLoader workers
  pin_memory: bool            # Pin memory for faster loading

# Training Configuration
training:
  epochs: int                 # Number of epochs
  learning_rate: float        # Initial learning rate
  weight_decay: float         # L2 regularization
  optimizer: str              # Optimizer (adam, sgd)
  loss: str                   # Loss function
  device: str                 # cuda or cpu
  seed: int                   # Random seed

# Augmentation Configuration
augmentation:
  enabled: bool               # Enable augmentation
  random_flip: bool           # Horizontal flip
  random_crop: bool           # Random crop
  color_jitter: bool          # Color jittering
  gaussian_blur: bool         # Gaussian blur
  rotation: float             # Rotation angle

# Checkpoint Configuration
checkpoint:
  save_interval: int          # Save every N epochs
  save_best: bool             # Save best model
  resume: str                 # Path to resume from
```

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
│
├── configs/                  # Configuration files
│   ├── cityscapes.yaml
│   ├── pascal_voc.yaml
│   └── custom.yaml
│
├── data/                     # Data handling modules
│   ├── __init__.py
│   ├── datasets.py           # Dataset classes
│   ├── loaders.py            # DataLoader utilities
│   └── augmentation.py       # Data augmentation
│
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── encoders/             # Encoder networks
│   ├── decoders/             # Decoder modules
│   ├── heads/                # Segmentation heads
│   └── backbones/            # Backbone networks
│
├── losses/                   # Loss functions
│   ├── __init__.py
│   ├── cross_entropy.py
│   ├── dice.py
│   └── focal.py
│
├── metrics/                  # Evaluation metrics
│   ├── __init__.py
│   ├── iou.py
│   ├── pixel_accuracy.py
│   └── confusion_matrix.py
│
├── training/                 # Training utilities
│   ├── __init__.py
│   ├── trainer.py            # Training loop
│   ├── validator.py          # Validation logic
│   └── callbacks.py          # Training callbacks
│
├── inference/                # Inference utilities
│   ├── __init__.py
│   ├── predictor.py          # Inference engine
│   ├── postprocess.py        # Post-processing
│   └── api.py                # REST API
│
├── scripts/                  # Utility scripts
│   ├── download_datasets.sh
│   ├── export_model.py
│   └── visualize_results.py
│
├── tests/                    # Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
│
├── checkpoints/              # Model checkpoints
├── results/                  # Training results
└── logs/                     # TensorBoard logs
```

## Advanced Features

### Custom Dataset Integration

To add a custom dataset:

```python
from data.datasets import SegmentationDataset

class CustomDataset(SegmentationDataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.images = []
        self.masks = []
        # Load your data here
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = load_image(self.images[idx])
        mask = load_mask(self.masks[idx])
        
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        return image, mask
```

### Custom Model Architecture

```python
import torch
import torch.nn as nn
from models import Encoder, Decoder

class CustomSegmentationModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        self.encoder = Encoder(backbone)
        self.decoder = Decoder(num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
```

### Using Tensorboard for Monitoring

```bash
# Start TensorBoard
tensorboard --logdir=logs/

# Access at http://localhost:6006
```

## Performance Benchmarks

| Model | Backbone | Dataset | mIoU (%) | FPS |
|-------|----------|---------|----------|-----|
| FCN | ResNet50 | Cityscapes | 71.2 | 28 |
| U-Net | ResNet50 | Cityscapes | 73.5 | 25 |
| DeepLabV3 | ResNet50 | Cityscapes | 79.3 | 20 |
| DeepLabV3+ | ResNet50 | Cityscapes | 80.1 | 18 |

*Note: Benchmarks are for reference only. Performance depends on hardware and implementation details.*

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
batch_size: 8  # instead of 16

# Use gradient accumulation
gradient_accumulation_steps: 2
```

### Slow Training

```bash
# Increase number of workers
num_workers: 8  # instead of 4

# Enable pin memory
pin_memory: true

# Use distributed training
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### Poor Model Performance

1. Check dataset quality and annotations
2. Increase training epochs
3. Adjust learning rate
4. Use stronger augmentation
5. Try different backbone architecture
6. Ensure proper class balancing

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Add docstrings to functions

## Citation

If you use this framework in your research, please cite:

```bibtex
@repository{generalized_semantic_segmentation,
  author = {Yoosofbidardel},
  title = {Generalizable Semantic Segmentation Framework},
  year = {2025},
  url = {https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV for computer vision utilities
- Dataset providers (Cityscapes, Pascal VOC, ADE20K, etc.)
- Community contributions and feedback

## Contact & Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Report bugs](https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation/issues)
- **Email**: [Your contact email]
- **Discussions**: [Join discussions](https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation/discussions)

## Related Resources

- [Semantic Segmentation Survey](https://arxiv.org/abs/1704.06857)
- [DeepLabV3+ Paper](https://arxiv.org/abs/1802.02611)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Last Updated**: December 31, 2025

Happy segmenting! 🎯

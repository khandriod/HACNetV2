# HACNetV2: Hybrid Attention-based Crack Detection Network

This repository contains the implementation of HACNetV2, a deep learning architecture for crack detection in images. The network uses a hybrid attention mechanism and parallel processing paths to achieve efficient and accurate crack detection.

## Project Structure

```
HACNetV2/
├── models/
│   ├── __init__.py
│   ├── hacnetv2.py
│   └── modules.py
├── utils/
│   ├── __init__.py
│   └── data_utils.py
├── config/
│   └── config.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

## Features

- Hybrid attention mechanism for crack detection
- Parallel processing paths for multi-scale feature extraction
- Efficient stem blocks for initial feature extraction
- Multi-branch output for different scales of crack detection

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HACNetV2.git
cd HACNetV2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config config/config.py
```

### Testing

```bash
python test.py --config config/config.py --checkpoint path/to/checkpoint.pth
```

## Model Architecture

The HACNetV2 architecture consists of:
- Effective stem blocks for initial feature extraction
- Hybrid ASPA (Attention-based Spatial Pyramid Attention) modules
- Multi-scale feature fusion
- Multi-branch output for different scales of crack detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# HACNetV2
HACNet V2: Rethinking the Full-Resolution  Network for Pixel-level Crack Detection
[paper](https://doi.org/10.1016/j.eswa.2025.128144)

Welcome to the official repository for HACNet V2, an advanced deep learning architecture designed for precise pixel-level crack detection. 

HACNetV2 is an enhanced full-resolution architecture that builds upon our previous HACNet. Through key architectural refinements and novel components, HACNetV2 achieves substantially improved efficiency and effectiveness compared to its predecessor. 

## 🔍 What distinguishes HACNetv2 from existing crack detection models?

### 🚀 Simplicity
A straightforward architecture composed of six identical HybridASPA blocks, each maintaining consistent structure, parameters, input/output channel dimensions, and resolution. 

### 🎯 Effectiveness
Superior detection accuracy, particularly in identifying tiny and fine-grained cracks.

### ⚡ Efficiency
With only **0.52M parameters**, achieves real-time inference performance in **Jetson Orin AGX**.


# The pixel-level Crack Datasets: 
## BCL dataset:
Harvard dataverse and can bevisited through the following link: https://doi.org/10.7910/DVN/RURXSH
## CHCrack5K dataset:
Github: https://github.com/hanshenChen/CHCrack5K


If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
# Citation
#### HACNetV2 Reference:
```bibtex
@article{CHEN2025128144,
title = {HACNet V2: Rethinking the full-resolution architecture for pixel-level crack detection},
journal = {Expert Systems with Applications},
pages = {128144},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.128144},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425017646},
author = {Hanshen Chen and Hao Chen}
}
```

#### HACNet Reference:
The source code for HACNet is available at [https://github.com/hanshenChen/HacNet](https://github.com/hanshenChen/HACNet).  
HACNet;
```bibtex
@ARTICLE{9410578,
  author={Chen, Hanshen and Lin, Huiping},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={An Effective Hybrid Atrous Convolutional Network for Pixel-Level Crack Detection}, 
  year={2021},
  volume={70},
  number={},
  pages={1-12},
  keywords={Feature extraction;Task analysis;Image segmentation;Spatial resolution;Maintenance engineering;Convolutional codes;Semantics;Atrous convolution;crack detection;defect inspection;image segmentation;neural network architecture},
  doi={10.1109/TIM.2021.3075022}}
```


# HACNetV2
HACNet V2: Rethinking the Full-Resolution  Network for Pixel-level Crack Detection

Welcome to the official repository for HACNet V2, an advanced deep learning architecture designed for precise pixel-level crack detection. 

HACNetV2 is an enhanced full-resolution architecture that builds upon our previous HACNet. Through key architectural refinements and novel components, HACNetV2 achieves substantially improved efficiency and effectiveness compared to its predecessor. 

## 🔍 What distinguishes HACNetv2 from existing crack detection models?

### 🚀 Simplicity
HACNetv2 employs a straightforward architecture composed of six identical HybridASPA blocks, each maintaining consistent structure, parameters, input/output channel dimensions, and resolution. 

### 🎯 Effectiveness
HACNetv2 demonstrates superior detection accuracy, particularly in identifying tiny and fine-grained cracks, outperforming existing state-of-the-art models.

### ⚡ Efficiency
With only **1.17M parameters**, HACNetv2 achieves real-time inference performance, making it highly suitable for deployment on resource-constrained devices such as the **Jetson Orin AGX**.

The source code for HACNet is available at [https://github.com/hanshenChen/HacNet](https://github.com/hanshenChen/HACNet).  

# The Datasets: 
## BCL dataset:
Harvard dataverse and can bevisited through the following link: https://doi.org/10.7910/DVN/RURXSH
## CHCrack5K dataset:
Github: https://github.com/hanshenChen/CHCrack5K


If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
# Citation
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



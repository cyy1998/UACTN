# UACTN
Official repository of Uncertainty-Aware Cross-Modal Transfer Network for Sketch-Based 3D Shape Retrieval (ICME 2023)

## Abstract
In recent years, sketch-based 3D shape retrieval has attracted growing attention. While many previous studies have focused on cross-modal matching between hand-drawn sketches and 3D shapes, the critical issue of how to handle low-quality and noisy samples in sketch data has been largely neglected. This paper presents an uncertainty-aware cross-modal transfer network (UACTN) that addresses this issue. UACTN decouples the representation learning of sketches and 3D shapes into two separate tasks: classification-based sketch uncertainty learning and 3D shape feature transfer. We first introduce an end-to-end classification-based approach that simultaneously learns sketch features and uncertainty, allowing uncertainty to prevent overfitting noisy sketches by assigning different levels of importance to clean and noisy sketches. Then, 3D shape features are mapped into the pre-learned sketch embedding space for feature alignment. Extensive experiments and ablation studies on two benchmarks demonstrate the superiority of our proposed method compared to state-of-the-art methods.


## Architecture
![Figure2](https://github.com/cyy1998/UACTN/assets/37933688/f022167e-94d7-4df4-a5c4-e978eb58f442)
The overall architecture of the proposed uncertainty-aware cross-modal transfer network (UACTN) for SBSR is illustrated. We decouple the task of cross-modal matching between sketches and 3D shapes into two separate learning tasks: (1) sketch data uncertainty learning, which aims to obtain a noise-robust sketch feature extraction model by introducing sketch uncertainty information into the training of a classification model; and (2) 3D shape feature transfer, where 3D shape features are mapped into the sketch embedding space under the guidance of sketch class centers. Finally, a cross-domain discriminative embedding space (i.e., sketches and 3D shapes belonging to the same class are close, while those of different classes are apart) is learned. The two tasks are discussed in detail in the following subsections.

## Qualitative Results

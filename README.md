# UACTN
Official repository of Uncertainty-Aware Cross-Modal Transfer Network for Sketch-Based 3D Shape Retrieval (ICME 2023)

## Abstract
In recent years, sketch-based 3D shape retrieval has attracted growing attention. While many previous studies have focused on cross-modal matching between hand-drawn sketches and 3D shapes, the critical issue of how to handle low-quality and noisy samples in sketch data has been largely neglected. This paper presents an uncertainty-aware cross-modal transfer network (UACTN) that addresses this issue. UACTN decouples the representation learning of sketches and 3D shapes into two separate tasks: classification-based sketch uncertainty learning and 3D shape feature transfer. We first introduce an end-to-end classification-based approach that simultaneously learns sketch features and uncertainty, allowing uncertainty to prevent overfitting noisy sketches by assigning different levels of importance to clean and noisy sketches. Then, 3D shape features are mapped into the pre-learned sketch embedding space for feature alignment. Extensive experiments and ablation studies on two benchmarks demonstrate the superiority of our proposed method compared to state-of-the-art methods.


## Architecture
![Figure2](https://github.com/cyy1998/UACTN/assets/37933688/f022167e-94d7-4df4-a5c4-e978eb58f442)
The overall architecture of the proposed uncertainty-aware cross-modal transfer network (UACTN) for SBSR is illustrated. We decouple the task of cross-modal matching between sketches and 3D shapes into two separate learning tasks: (1) sketch data uncertainty learning, which aims to obtain a noise-robust sketch feature extraction model by introducing sketch uncertainty information into the training of a classification model; and (2) 3D shape feature transfer, where 3D shape features are mapped into the sketch embedding space under the guidance of sketch class centers. Finally, a cross-domain discriminative embedding space (i.e., sketches and 3D shapes belonging to the same class are close, while those of different classes are apart) is learned. The two tasks are discussed in detail in the following subsections.

## Code
A workable basic version of the code for CLIP adapted for ZS-SBIR has been uploaded.
- ```train_sketch.py``` python script to train the sketch model.
- ```train_view.py``` python script to train the view model.
- ```retrieval_evaluation.py``` python script to run the experiment.

## Qualitative Results

Qualitative results of ZS-SBIR on Sketchy by a baseline (blue) method vs Ours (green).
![Figure5](https://github.com/cyy1998/UACTN/assets/37933688/03f4ca3b-c69d-43c5-bc3f-933ade2f9be0)

## Quantitative Results
Quantitative results of our method against a few SOTAs.
![图片](https://github.com/cyy1998/UACTN/assets/37933688/2dc21309-eff1-4f49-ae95-5fd49100b0b9)

## Bibtex
Please cite our work if you found it useful. Thanks.
```
@INPROCEEDINGS{10219630,
  author={Cai, Yiyang and Lu, Jiaming and Wang, Jiewen and Liang, Shuang},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Uncertainty-Aware Cross-Modal Transfer Network for Sketch-Based 3D Shape Retrieval}, 
  year={2023},
  volume={},
  number={},
  pages={132-137},
  keywords={Representation learning;Learning systems;Three-dimensional displays;Uncertainty;Shape;Benchmark testing;Data models;sketch;3D shape retrieval;data uncertainty learning},
  doi={10.1109/ICME55011.2023.00031}}
```  


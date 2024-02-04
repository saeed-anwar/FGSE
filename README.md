# A Systematic Evaluation: Fine-Grained CNN vs. Traditional CNN Classifiers
This repository is for comparison of fine-grain classifiers and traditional classifiers introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/), Nick Barnes, and Lars Petersson, "[A Systematic Evaluation: Fine-Grained CNN vs. Traditional CNN Classifiers](https://arxiv.org/pdf/2003.11154.pdf)", Electronics, 2023

## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Requirements](#requirements)
4. [Test](#test)
5. [Results](#results)
6. [Citation](#citation)

## Introduction
To make the best use of the underlying minute and subtle differences, fine-grained classifiers collect information about inter-class variations.  The task is very challenging due to the small differences between the colors, viewpoint, and structure in the same class entities. The classification becomes more difficult due to the similarities between the differences in viewpoint with other classes and differences with its own.  In this work, we investigate the performance of the landmark general CNN classifiers, which presented top-notch results on large scale classification datasets, on the fine-grained datasets, and compare it against state-of-the-art fine-grained classifiers. This paper poses two specific questions: (i) Do the general CNN classifiers achieve comparable results to fine-grained classifiers? (ii) Do general CNN classifiers require any specific information to improve upon the fine-grained ones? We train the general CNN classifiers throughout this work without introducing any aspect specific to fine-grained datasets. We show an extensive evaluation on six datasets to determine whether the fine-grained classifier can elevate the baseline in their experiments.

The difference between classes (inter-class variation) is limited forvarious classes
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/Fine1.png">
</p>

The intra-class variation is usually high due to pose, lighting, and color.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/Fine2.png">
</p>


## Network

The networks in this evaluation can be broadly categorized into plain networks, residual networks, and densely connected networks. The three architectures we investigate against fine-grained classifiers are VGG, ResNet, and DenseNet. 

## Requirements

The model is built using
  1. PyTorch 1.6.0
  2. Python 3.8
  3. Cuda 10.2
  4. imageio
  5. logging
  6. Tested on Ubuntu 14.04/16.04 environment 

## Test
### Quick start
1. Download the trained models from

   - [Airplane](https://drive.google.com/file/d/1Fx-KvAl-PZzAVZr-i96Y0ab0bQ81U8kS/view?usp=sharing) (**401MB** in size.)
   - [Cars](https://drive.google.com/file/d/1v-N7WFQ15nOKw-boHTw7Od6q0CataWZY/view?usp=sharing)  (**404MB** in size.)
   - [CUB](https://drive.google.com/file/d/1F36YjAsb7dQ-zIGqcF0MobkzXNY-8FNJ/view?usp=sharing)  (**403MB** in size.)
   - [Dogs](https://drive.google.com/file/d/1m4x95JHBx4nHHQjEzjeVBvI7pEsp3eOJ/view?usp=sharing)  (**402MB** in size.)
   - [Flowers](https://drive.google.com/file/d/1EvjTo707DNER3sKqOjspnf2ApzHWTptr/view?usp=sharing) (**401MB** in size.)
   - [NaBirds](https://drive.google.com/file/d/1LIg_AxTHTmsCLmTW9WnJPp5XN8whSwxx/view?usp=sharing) (**414MB** in size.)


2. cd to the dataset such Airplane, Cars, CUB, Dogs, Flowers and NaBirds and then cd to the model r50, r152 or d161
   for example, '/Cars/r50_cars/' or '/Airplane/r152_plane', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #Script
    CUDA_VISIBLE_DEVICES=0 python test.py 
    ```

## Results
### Datasets
Details of six fine-grained visual categorization datasets to evaluate the proposed method.
<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/Datasets.png">
</p>

### Quantitative Results

Performance comparison with Traditional CNN learning and Fine-grained algorithms. Comparison of the state-of-the-art fine grain classification on
CUB dataset. Best results are highlighted in bold.
<p align="center">
  <img width="300" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/CUB.png">
</p>


Experimental results on FGVC Aircraft and Cars.
<p align="center">
  <img width="400" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/Cars.png">
</p>

Comparison of the state-of-the-art fine grain classification on Dogs, Flowers, and NABirds dataset.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/threeDatasets.png">
</p>

Differences strategies for initialing the network weights i.e. finetuning from ImageNet and random initialization (scratch) for Cars dataset.
<p align="center">
  <img width="250" src="https://github.com/saeed-anwar/FGSE/blob/master/Figs/Ablation.png">
</p>


For more information, please refer to our [paper](https://arxiv.org/pdf/2003.11154.pdf)

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{anwar2020systematic,
  title={A Systematic Evaluation: Fine-Grained CNN vs. Traditional CNN Classifiers},
  author={Anwar, Saeed and Barnes, Nick and Petersson, Lars},
  journal={Electronics},
  year={2023}
  volume = {12},
  number = {23},
  article-number = {4877},
  issn = {2079-9292}
}

```

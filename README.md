# A Systematic Evaluation: Fine-Grained CNN vs. Traditional CNN Classifiers
This repository is for comparison of fine-grain classifiers and traditional classifiers introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/), Nick Barnes, and Lars Petersson, "[A Systematic Evaluation: Fine-Grained CNN vs. Traditional CNN Classifiers](https://arxiv.org/pdf/2003.11154.pdf)" 

## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Requirements](#requirements)
4. [Test](#test)
5. [Results](#results)
6. [Citation](#citation)

## Introduction
To make the best use of the underlying minute and subtle differences, fine-grained classifiers collect information about inter-class variations.  The task is very challenging due to the small differences between the colors, viewpoint, and structure in the same class entities. The classification becomes more difficult due to the similarities between the differences in viewpoint with other classes and differences with its own.  In this work, we investigate the performance of the landmark general CNN classifiers, which presented top-notch results on large scale classification datasets, on the fine-grained datasets, and compare it against state-of-the-art fine-grained classifiers. This paper poses two specific questions: (i) Do the general CNN classifiers achieve comparable results to fine-grained classifiers? (ii) Do general CNN classifiers require any specific information to improve upon the fine-grained ones? We train the general CNN classifiers throughout this work without introducing any aspect specific to fine-grained datasets. We show an extensive evaluation on six datasets to determine whether the fine-grained classifier can elevate the baseline in their experiments.

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Example_images.png">
</p>
Image datasets for protein localization; each image belongs to a different class. Most of the images
are sparse.

## Network

The architecture of the proposed network. A glimpse of the proposed network used for localization of the protein structures in the cell. The
composition of R_s, R_l, P_s and P_l are provided below the network structure, where the subscript s have a small number of convolutions as compared to l

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Network.png">
</p>

## Requirements

The model is built using
  1. PyTorch 1.5.1
  2. Python 3.8
  3. Cuda 10.2
  4. Tested on Ubuntu 14.04/16.04 environment 

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The PLCNN model can be downloaded from [Google Drive]() or [here](). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #Script
    CUDA_VISIBLE_DEVICES=0 python main.py 
    ```

## Results
**All the results for HeLa, CHO, Endo, Trans and Yeast.** 

### Quantitative Results

Performance comparison with machine learning and CNN-Specific algorithms. The “Endo” and “Trans” is the abbreviation for LOCATE Endogenous and Transfected datasets, respectively. Best results are highlighted in bold.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/CNNvsMachine.png">
</p>


Performance against traditional CNN methods using Yeast and HeLa datasets. The best results are in bold.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/PLCNNvsTCNN.png">
</p>

The effect of decreasing the training dataset. It can be observed that the performance decrease for traditional ensemble algorithms with the decrease in training data while, on the other hand, PLCNN gives a consistent performance with a negligible difference.
<p align="center">
  <img width="400" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/EffectDecreasingTraining.png">
</p>

ETAS accuracies for individual members of ensemble on CHO dataset for tau = 40.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/ETAS_accuracy.png">
</p>


For more information, please refer to our [papar](https://arxiv.org/pdf/1910.04287.pdf)

### Confusion matrices 
**The confusion matrices for different datasets.**

Confusion matrix for CHO dataset. The rows present the actual organelle class while the columns show the predicted ones. The results are aggregated for 10-fold cross-validations. The accuracies for each class are summarized in the last row as well as columns.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/CHO_CM.png">
</p>

Confusion matrix for Yeast dataset. The predicted organelle are shown in the columns while the true values are present in the rows. The summaries of accuracies are given in the last row and column.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/DeepYeast_CM.png">
</p>


The correct predictions are highlighted via green while the red depicts incorrect. Our method prediction score is high for true outcome and vice versa.
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Correctpredict.png">
</p>

The average quantitative results of ten execution for each method on the HeLa dataset. Our PLCNN method consistently outperforms with a significant margin.
<p align="center">
  <img width="400" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/PLCNNvsTCNNGraph.png">
</p>


Visualization results from Grad-CAM. The visualization is computed for the last convolutional outputs, and the corresponding algorithms are shown in the left column the input images
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/CAM1.png">
</p>
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/CAM2.png">
</p>

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{anwar2020systematic,
  title={A Systematic Evaluation: Fine-Grained CNN vs. Traditional CNN Classifiers},
  author={Anwar, Saeed and Barnes, Nick and Petersson, Lars},
  journal={arXiv preprint arXiv:2003.11154},
  year={2020}
}

```

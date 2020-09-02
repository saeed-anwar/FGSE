# Deep localization of protein structures in fluorescence microscopy images
This repository is for Deep localization of protein structures in fluorescence microscopy images (PLCNN) introduced in the following paper

Muhammad Tahir, [Saeed Anwar](https://saeed-anwar.github.io/), and [Ajmal Mian](https://research-repository.uwa.edu.au/en/persons/ajmal-mian), "[Deep localization of protein structures in fluorescence microscopy images](https://arxiv.org/abs/1910.04287)" 

The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1). 


## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Accurate localization of proteins from fluorescence microscopy images is a challenging task due to the inter-class similarities and intra-class disparities introducing grave concerns in addressing multi-class classification problems. Conventional machine learning-based image prediction relies heavily on pre-processing such as normalization and segmentation followed by hand-crafted feature extraction before classification to identify useful and informative as well as application specific features.We propose an end-to-end Protein Localization Convolutional Neural Network (PLCNN) that classifies protein localization images more accurately and reliably. PLCNN directly processes raw imagery without involving any pre-processing steps and produces outputs without any customization or parameter adjustment for a particular dataset. The output of our approach is computed from probabilities produced by the network. Experimental analysis is performed on five publicly available benchmark datasets. PLCNN consistently outperformed the existing state-of-the-art approaches from machine learning and deep architectures.

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

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The PLCNN model can be downloaded from [Google Drive]() or [here](). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #RIDNET
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model RIDNET --n_feats 64 --pre_train ../experiment/ridnet.pt --test_only --save_results --save 'RIDNET_RNI15' --testpath ../LR/LRBI/ --testset RNI15
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
@article{tahir2019PLCNN,
  title={Deep localization of protein structures in fluorescence microscopy images},
  author={Tahir, Muhammad and Anwar, Saeed and Mian, Ajmal},
  journal={arXiv preprint arXiv:1910.04287},
  year={2019}
}

```

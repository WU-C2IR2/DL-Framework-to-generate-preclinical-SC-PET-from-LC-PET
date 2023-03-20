# Deep Learning Framework for generating preclinical SC-PET images from LC-PET realizations
This repository provides the code for a deep-learning based framework for generation and multi-objective evaluation of preclinical Standard-Count PET (SC-PET) images from different realizations of Low-Count PET (LC-PET) images. To that end in this project we have developed and optimized a novel deep-learning architecture called Attention based Residual Dilated Network (ARD-Net). The performance of ARD-Net generated SC-PET images were benchmarked against other deep-learning methods i.e. Residual UNet (RU-Net) and Dilated Net (D-Net) and non deep-learning denoisers like Non-Local Means (NLM) and Block Matching and 3D Filtering (BM3D).

## Steps to Train the DL framework for SC-PET generation
1. Git Clone the repository
2. Install `python 3.7.10` and necessary packages by running `conda create --name <env> --file requirements.txt` or by running the `tf_37.yml` file. 
3. Download the dataset folder `Data_Preclinical` using the link into the same folder where you have cloned the other codes from the repository. The dataset has two subfolders namely: `Training` and `Testing`. The `Training` folder consists of the training samples both in the raw data format and in the .mat format for the different photon count levels. Similarly `Testing` consists of the independent testing dataset for the evaluation of the framework.
4. Training the network:<br>
  a) Choose the filepath for the training data. <br>
  b) Choose the hyperparameters. <br>
  c) Run the main file of the network which you intend to execute by using the command `python main_ardnet.py --args train` would run the training for ARD-Net using the training dataset for the desired number of epochs. <br>
  



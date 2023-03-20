# Deep Learning Framework for generating preclinical SC-PET images from LC-PET realizations
This repository provides the code for a deep-learning based framework for generation and multi-objective evaluation of preclinical Standard-Count PET (SC-PET) images from different realizations of Low-Count PET (LC-PET) images. To that end in this project we have developed and optimized a novel deep-learning architecture called Attention based Residual Dilated Network (ARD-Net). The performance of ARD-Net generated SC-PET images were benchmarked against other deep-learning methods i.e. Residual UNet (RU-Net) <sup>1</sup> and Dilated Net (D-Net) <sup>2</sup> and non deep-learning denoisers like Non-Local Means (NLM) <sup>3</sup> and Block Matching and 3D Filtering (BM3D) <sup>4</sup>.

## Steps to Train the DL framework for SC-PET generation
1. Git Clone the repository
2. Install `python 3.7.10` and necessary packages by running `conda create --name <env> --file requirements.txt` or by running the `tf_37.yml` file. 
3. Download the dataset folder `Data_Preclinical` using the link into the same folder where you have cloned the other codes from the repository. The dataset has two subfolders namely: `Training` and `Testing`. The `Training` folder consists of the training samples both in the raw data format and in the .mat format for the different photon count levels. Similarly `Testing` consists of the independent testing dataset for the evaluation of the framework.
4. Training the network:<br>
  a) Choose the filepath for the training data. <br>
  b) Choose the hyperparameters. <br>
  c) Run the main file of the network which you intend to execute by using the command `python main_ardnet.py --args train` would run the training for ARD-Net using the training dataset for the desired number of epochs. <br>
5. Testing the network:<br>
  a) Choose the filepath for the testing data. <br>
  b) Run the main file of the network which you intend to execute by using the command `python main_ardnet.py --args test` would run the testing for ARD-Net using the testing dataset.<br>

### Different Network Model and their architecture files:
| Network Model | Architecture File | Main File |
| --- | --- | --- |
| ARD-Net | ard-net.py | main_ardnet.py |
| D-Net | dilated_unet.py | main_dnet.py |
| RU-Net | residual_unet.py | main_runet.py |


### References
1. He, K.;  Zhang, X.;  Ren, S.; Sun, J., Deep residual learning for image recognition. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2016, 2016-Decem, 770-778.
2. Yu, F., Koltun, V., & Funkhouser, T. (2017). Dilated residual networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 472-480).
3. Buades, A., Coll, B., & Morel, J. M. (2011). Non-local means denoising. Image Processing On Line, 1, 208-212.
4. Lebrun, M. (2012). An analysis and implementation of the BM3D image denoising method. Image Processing On Line, 2012, 175-213.


  



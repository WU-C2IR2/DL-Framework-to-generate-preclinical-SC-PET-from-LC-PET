# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:29:34 2021

@author: kaushik.dutta
"""

############# THIS SCRIPT IS FOR CALLING THE DILATED-NET ###############


import math
from natsort import natsorted
from dilated_unet import denoisenet
from data_read import load_train_data, load_test_data, load_validation_data, name_selection
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from sklearn.model_selection import GridSearchCV
import argparse
import matplotlib

########### Accessing two GPUS #################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest = 'epoch', type = int, default = 250, help='No of Epochs')
parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 10, help = 'Batch Size')
parser.add_argument('--loss', dest = 'loss', default = 'mse', help = 'Loss Function')
parser.add_argument('--phase', dest = 'phase', default = 'train', help = 'Train and Test')
parser.add_argument('--filters', dest = 'filters', type = int, default = 64, help = 'Filter Size')
parser.add_argument('--time', dest = 'time', type = int, default = 1, help = 'Time')
args = parser.parse_args()

name = 'dnet_weights_'+ '_Loss_'+ str(args.loss) + '_BatchSize_' + str(args.batch_size) + 'filter_size' + str(args.filters) + '_epoch_' + 'time_lc = ' + str(args.time)
output_name = 'output/'+'dnet_weights_'+ '_Loss_'+ str(args.loss) + '_BatchSize_' + str(args.batch_size) + '_epoch_'

############# Adaptive Learning Rate Function ##################
def step_decay(epoch):
   initial_lrate = 1e-4
   drop = 0.5
   epochs_drop = 25.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def train():
    print('=============Loading of Training Data and Preprocessing==============')
    hc_train, lc_train = load_train_data()
    hc_train = hc_train.astype('float32')
    lc_train = lc_train.astype('float32')
    hc_validation, lc_validation = load_validation_data()
    hc_validation = hc_validation.astype('float32')
    lc_validation = lc_validation.astype('float32')
    
    loss_func = args.loss
    filter_size = args.filters
    model = denoisenet(filter_size)
    weight_directory = 'weights'
    if not os.path.exists(weight_directory):
        os.mkdir(weight_directory)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_directory, name + '{epoch:04d}.h5'), monitor = 'loss', verbose = 0, save_best_only=True, period = 50)
    
    
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    logger = CSVLogger(os.path.join(log_directory,'log.csv'), separator = ',', append = False)
    early_stopping = EarlyStopping(monitor='loss', patience=15)
    learning_rate = LearningRateScheduler(step_decay)
    history = model.fit(lc_train, hc_train, batch_size = args.batch_size, epochs = args.epoch, callbacks = [model_checkpoint, logger, learning_rate], validation_data = (lc_validation, hc_validation), validation_batch_size = 1)
    scores = model.evaluate(lc_validation, hc_validation)
    ssim = scores[2]
    print('===============Training Done==========================')
    file = open("performance_dnet1.txt", "a")
    file.write("Batch_Size = " + str(args.batch_size) +  "  Loss Function = " + str(args.loss) +  "  Initial Filter Size = " + str(args.filters) + " Time_lc = " + str(args.time) + "  SSIM = " + str(ssim) + "\n")
    file.close()
    print(ssim)
    
    
    
def predict():
    print('==========Loading Of testing Data =====================')
    hc_gt, lc_test = load_test_data()
    lc_test = lc_test.astype('float32')
    
    hc_gt = hc_gt.astype('float32')
    weight_directory = 'weights'
    model = denoisenet(str(loss_func), filter_size)
    model.load_weights(os.path.join(weight_directory,output_name + '0050.h5'))
    hc_pred = model.predict(lc_test, batch_size = 1, verbose = 1)
    return hc_gt, hc_pred, lc_test
    
if __name__ == '__main__':

    if args.phase == 'train':
        train()
    elif args.phase == 'tests':
        gt, pred, gt_lc = predict()
        new_gt = np.mean(gt, axis = 3)
        new_pred = np.mean(pred, axis = 3)
        new_gt_lc = np.mean(gt_lc, axis = 3)
        np.save('output/' + output_name + '/'+'gt.npy',gt)
        np.save('output/' + output_name + '/'+'gt_lc.npy',gt_lc)
        np.save('output/' + output_name + '/'+'pred.npy',pred)
        
        ssim_calc_pred = np.zeros((len(new_gt),1))
        ssim_calc_lc = np.zeros((len(new_gt),1))
        psnr_calc_pred = np.zeros((len(new_gt),1))
        psnr_calc_lc = np.zeros((len(new_gt),1))
        nrmse_calc_pred = np.zeros((len(new_gt),1))
        nrmse_calc_lc = np.zeros((len(new_gt),1))
        
        for i in range(0,len(new_gt)):
            max_val = np.max(new_gt[i,:,:])
            ssim_calc_pred[i] = ssim(new_gt[i,:,:], new_pred[i,:,:], data_range = new_pred[i,:,:].max() - new_pred[i,:,:].min())
            ssim_calc_lc[i] = ssim(new_gt[i,:,:], new_gt_lc[i,:,:])
            psnr_calc_pred[i] = psnr(new_gt[i,:,:], new_pred[i,:,:], data_range = new_pred[i,:,:].max() - new_pred[i,:,:].min())
            psnr_calc_lc[i] = psnr(new_gt[i,:,:], new_gt_lc[i,:,:], data_range = max_val)
            nrmse_calc_pred[i] = nrmse(new_gt[i,:,:], new_pred[i,:,:], normalization='euclidean')
            nrmse_calc_lc[i] = nrmse(new_gt[i,:,:], new_gt_lc[i,:,:], normalization='euclidean')
        
        print('SSIM for DL Method     Mean = ', np.mean(ssim_calc_pred), 'Standard Deviation =', np.std(ssim_calc_pred))
        print('SSIM for LC Method     Mean = ', np.mean(ssim_calc_lc), 'Standard Deviation =', np.std(ssim_calc_lc))
        
        print('PSNR for DL Method     Mean = ', np.mean(psnr_calc_pred), 'Standard Deviation =', np.std(psnr_calc_pred))
        print('PSNR for LC Method     Mean = ', np.mean(psnr_calc_lc), 'Standard Deviation =', np.std(psnr_calc_lc))
        
        print('NRMSE for DL Method     Mean = ', np.mean(nrmse_calc_pred), 'Standard Deviation =', np.std(nrmse_calc_pred))
        print('NRMSE for LC Method     Mean = ', np.mean(nrmse_calc_lc), 'Standard Deviation =', np.std(nrmse_calc_lc))

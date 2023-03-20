# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:37:53 2022

@author: kaushik.dutta
"""

######### THIS PYTHON SCRIPT READS DATA AND PREPARES THE TENSOR SO THAT CAN BE UTILIZED FOR DEEP LEARNING FOR DIFFERENT PHOTON COUNT REALIZATIONS #####################

import numpy as np
import os
import natsort
import scipy.io as sio
from sklearn.model_selection import train_test_split
 
def name_selection(time):
    if time==1:
        str_name = str(time)+'mins_mat/'
    elif time==30:
        str_name = str(time)+'secs_mat/'
    elif time==10:
        str_name = str(time)+'secs_mat/'
    elif time==5:
        str_name = str(time)+'secs_mat/'
    
    return str_name
        
############## Reading the Standard Count and Low Count Data ######
hc_path_train = 'data/Training/mat_files/10mins_mat/'
lc_path_train = 'data/Training/mat_files/' + name_selection(time) + '/'
hc_list_train = natsort.natsorted(os.listdir(hc_path_train))
lc_list_train = natsort.natsorted(os.listdir(lc_path_train))

hc_path_test = 'data/Testing/mat_files/10mins_mat/'
lc_path_test = 'data/Testing/mat_files/' + name_selection(time) + '/'
hc_list_test = natsort.natsorted(os.listdir(hc_path_test))
lc_list_test = natsort.natsorted(os.listdir(lc_path_test))

hc_train, hc_validation, lc_train, lc_validation = train_test_split(hc_list_train, lc_list_train, test_size = 0.15, random_state = 42)
hc_train = natsort.natsorted(hc_train)
lc_train = natsort.natsorted(lc_train)
hc_validation = natsort.natsorted(hc_validation)
lc_validation = natsort.natsorted(lc_validation)


for i in range(0, len(hc_train)):
    new_hc_data_train_1 = sio.loadmat(hc_path_train + hc_train[i])['hc']
    new_lc_data_train_1 = sio.loadmat(lc_path_train + lc_train[i])['lc']
    
    hc_data_train_1 = new_hc_data_train_1[:,:,10:145]
    lc_data_train_1 = new_lc_data_train_1[:,:,10:145]
    
    if i==0:
        hc_data_final = hc_data_train_1
        lc_data_final = lc_data_train_1
    else:
        hc_data_final = np.dstack((hc_data_final, hc_data_train_1))
        lc_data_final = np.dstack((lc_data_final, lc_data_train_1))

for i in range(0, len(hc_list_test)):
    new_hc_data_test_1 = sio.loadmat(hc_path_test + hc_list_test[i])['hc']
    new_lc_data_test_1 = sio.loadmat(lc_path_test + lc_list_test[i])['lc']
    
    hc_data_test_1 = new_hc_data_test_1[:,:,10:145]
    lc_data_test_1 = new_lc_data_test_1[:,:,10:145]
    
    if i==0:
        hc_data_test_final = hc_data_test_1
        lc_data_test_final = lc_data_test_1
        
    else:
        hc_data_test_final = np.dstack((hc_data_test_final, hc_data_test_1))
        lc_data_test_final = np.dstack((lc_data_test_final, lc_data_test_1))
        

for i in range(0, len(hc_validation)):
    new_hc_data_validation_1 = sio.loadmat(hc_path_train + hc_validation[i])['hc']
    new_lc_data_validation_1 = sio.loadmat(lc_path_train + lc_validation[i])['lc']
    
    hc_data_validation_1 = new_hc_data_validation_1[:,:,10:145]
    lc_data_validation_1 = new_lc_data_validation_1[:,:,10:145]
    
    if i==0:
        hc_data_validation_final = hc_data_validation_1
        lc_data_validation_final = lc_data_validation_1
        
    else:
        hc_data_validation_final = np.dstack((hc_data_validation_final, hc_data_validation_1))
        lc_data_validation_final = np.dstack((lc_data_validation_final, lc_data_validation_1))
        

############ Preparing the Data Tensor for Deep-learning ####################        
hc_data_final = np.expand_dims(hc_data_final, axis=3)
lc_data_final = np.expand_dims(lc_data_final, axis = 3)
hc_data_test_final = np.expand_dims(hc_data_test_final, axis=3)
lc_data_test_final = np.expand_dims(lc_data_test_final, axis = 3)
hc_data_validation_final = np.expand_dims(hc_data_validation_final, axis=3)
lc_data_validation_final = np.expand_dims(lc_data_validation_final, axis = 3)

hc_data_final = np.swapaxes(hc_data_final,0,2)
lc_data_final = np.swapaxes(lc_data_final,0,2)
hc_data_test_final = np.swapaxes(hc_data_test_final,0,2)
lc_data_test_final = np.swapaxes(lc_data_test_final,0,2)
hc_data_validation_final = np.swapaxes(hc_data_validation_final,0,2)
lc_data_validation_final = np.swapaxes(lc_data_validation_final,0,2)


############ Normalizing the intensities #########################
hc_data_final = hc_data_final/10000
lc_data_final = lc_data_final/10000
hc_data_test_final = hc_data_test_final/10000
lc_data_test_final = lc_data_test_final/10000
hc_data_validation_final = hc_data_validation_final/10000
lc_data_validation_final = lc_data_validation_final/10000


def load_train_data():
    return hc_data_final, lc_data_final

def load_test_data():
    return hc_data_test_final, lc_data_test_final
    
def load_validation_data():
    return hc_data_validation_final, lc_data_validation_final

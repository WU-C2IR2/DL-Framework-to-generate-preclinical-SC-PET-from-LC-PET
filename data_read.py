# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:05:03 2021

@author: kaushik.dutta
"""
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import scipy.io as sio
from natsort import natsorted

hc_path = '/home/kslabuser1/Data/op_hc_10mins/new/'
lc_path = '/home/kslabuser1/Data/op_lc_30secs/new/'

hc_list = os.listdir(hc_path)
lc_list = os.listdir(lc_path)

hc_list = natsorted(hc_list)
lc_list = natsorted(lc_list)

hc_train_1, hc_test_1, lc_train_1, lc_test_1 = train_test_split(hc_list, lc_list, test_size = 0.20, random_state=42)
hc_train_1, hc_validation_1, lc_train_1, lc_validation_1 = train_test_split(hc_train_1, lc_train_1, test_size = 0.10, random_state=42)
hc_train_1 = natsorted(hc_train_1)
hc_test_1 = natsorted(hc_test_1)
lc_train_1 = natsorted(lc_train_1)
lc_test_1 = natsorted(lc_test_1)
hc_validation_1 = natsorted(hc_validation_1)
lc_validation_1 = natsorted(lc_validation_1)

# print(hc_train_1)
# print(lc_train_1)

hc_data = np.ndarray((128,128,126))
lc_data = np.ndarray((128,128,126))
hc_data_test = np.ndarray((128,128,126))
lc_data_test = np.ndarray((128,128,126))
hc_data_validation = np.ndarray((128,128,126))
lc_data_validation = np.ndarray((128,128,126))

for i in range(0, len(hc_train_1)):
    new_hc_data_train_1 = sio.loadmat(hc_path + hc_train_1[i])['hc']
    new_lc_data_train_1 = sio.loadmat(lc_path + lc_train_1[i])['lc']
    
    hc_data_train_1 = new_hc_data_train_1[:,:,10:145]
    lc_data_train_1 = new_lc_data_train_1[:,:,10:145]
    
    if i==0:
        hc_data_final = hc_data_train_1
        lc_data_final = lc_data_train_1
    else:
        hc_data_final = np.dstack((hc_data_final, hc_data_train_1))
        lc_data_final = np.dstack((lc_data_final, lc_data_train_1))

for i in range(0, len(hc_test_1)):
    new_hc_data_test_1 = sio.loadmat(hc_path + hc_test_1[i])['hc']
    new_lc_data_test_1 = sio.loadmat(lc_path + lc_test_1[i])['lc']
    
    hc_data_test_1 = new_hc_data_test_1[:,:,10:145]
    lc_data_test_1 = new_lc_data_test_1[:,:,10:145]
    
    if i==0:
        hc_data_test_final = hc_data_test_1
        lc_data_test_final = lc_data_test_1
        
    else:
        hc_data_test_final = np.dstack((hc_data_test_final, hc_data_test_1))
        lc_data_test_final = np.dstack((lc_data_test_final, lc_data_test_1))
        

for i in range(0, len(hc_validation_1)):
    new_hc_data_validation_1 = sio.loadmat(hc_path + hc_validation_1[i])['hc']
    new_lc_data_validation_1 = sio.loadmat(lc_path + lc_validation_1[i])['lc']
    
    hc_data_validation_1 = new_hc_data_validation_1[:,:,10:145]
    lc_data_validation_1 = new_lc_data_validation_1[:,:,10:145]
    
    if i==0:
        hc_data_validation_final = hc_data_validation_1
        lc_data_validation_final = lc_data_validation_1
        
    else:
        hc_data_validation_final = np.dstack((hc_data_validation_final, hc_data_validation_1))
        lc_data_validation_final = np.dstack((lc_data_validation_final, lc_data_validation_1))

    
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

hc_data_final = hc_data_final/10000
lc_data_final = lc_data_final/10000
hc_data_test_final = hc_data_test_final/10000
lc_data_test_final = lc_data_test_final/10000
hc_data_validation_final = hc_data_validation_final/10000
lc_data_validation_final = lc_data_validation_final/10000

# lc_data_final[lc_data_final<1500] = 0
# hc_data_final[hc_data_final<1500] = 0
# lc_data_test_final[lc_data_test_final<1500] = 0
# hc_data_test_final[hc_data_test_final<1500] = 0

def load_train_data():
    return hc_data_final, lc_data_final

def load_test_data():
    return hc_data_test_final, lc_data_test_final
    
def load_validation_data():
    return hc_data_validation_final, lc_data_validation_final

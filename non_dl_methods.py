# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 07:09:44 2022

@author: kaushik.dutta
"""
import numpy as np
from matplotlib import pyplot as plt
from read_data_singleslice import load_train_data, load_test_data
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
import bm3d

hc_gt, lc_test = load_test_data()
lc_test = lc_test.astype('float32')
hc_gt = hc_gt.astype('float32')

lc_test = np.mean(lc_test, axis = 3)
hc_gt = np.mean(hc_gt, axis = 3)

hc_gt = np.swapaxes(hc_gt,0,2)
lc_test = np.swapaxes(lc_test,0,2)
no_slices = lc_test.shape[2]

nlm_denoise = denoise_nl_means(lc_test, fast_mode = True, multichannel = False)
bm3d_denoise = bm3d.bm3d(lc_test, sigma_psd=30/255)

ssim_calc_nlm = np.zeros((no_slices,1))
ssim_calc_bm3d = np.zeros((no_slices,1))
psnr_calc_nlm = np.zeros((no_slices,1))
psnr_calc_bm3d = np.zeros((no_slices,1))
nrmse_calc_nlm = np.zeros((no_slices,1))
nrmse_calc_bm3d = np.zeros((no_slices,1))

for i in range(0,no_slices):
    max_val = np.max(hc_gt[:,:,i])
    ssim_calc_nlm[i] = ssim(hc_gt[:,:,i], nlm_denoise[:,:,i])
    ssim_calc_bm3d[i] = ssim(hc_gt[:,:,i], bm3d_denoise[:,:,i])
    psnr_calc_nlm[i] = psnr(hc_gt[:,:,i], nlm_denoise[:,:,i], data_range = max_val)
    psnr_calc_bm3d[i] = psnr(hc_gt[:,:,i], bm3d_denoise[:,:,i], data_range = max_val)
    nrmse_calc_nlm[i] = nrmse(hc_gt[:,:,i], nlm_denoise[:,:,i], normalization='euclidean')
    nrmse_calc_bm3d[i] = nrmse(hc_gt[:,:,i], bm3d_denoise[:,:,i], normalization='euclidean')

ssim_nlm_mean = np.mean(ssim_calc_nlm)
ssim_nlm_std = np.std(ssim_calc_nlm)
ssim_bm3d_mean = np.mean(ssim_calc_bm3d)
ssim_bm3d_std = np.std(ssim_calc_bm3d)

psnr_nlm_mean = np.mean(psnr_calc_nlm)
psnr_nlm_std = np.std(psnr_calc_nlm)
psnr_bm3d_mean = np.mean(psnr_calc_bm3d)
psnr_bm3d_std = np.std(psnr_calc_bm3d)

nrmse_nlm_mean = np.mean(nrmse_calc_nlm)
nrmse_nlm_std = np.std(nrmse_calc_nlm)
nrmse_bm3d_mean = np.mean(nrmse_calc_bm3d)
nrmse_bm3d_std = np.std(nrmse_calc_bm3d)


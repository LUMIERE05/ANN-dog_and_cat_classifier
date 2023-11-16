# MEMANGGIL LIBRARY #
import numpy as np
import os, sys
import warnings
from numpy.random import randn
from sklearn.preprocessing import MinMaxScaler
#########################

warnings.filterwarnings('ignore')

# HYPER PARAMETER #
jumlah_data = 1000
jumlah_output = 2
jumlah_input = 1
jumlah_neuron_hidden = 1000
gambar_data = int(jumlah_data / 2)
nfeature1 = 10000
nfeature2 = 100
panjang_data = jumlah_output * jumlah_data  
keterangan_background = "ada" # ada or tanpa  
keterangan_data       = "training"    # training OR testing  
percobaan_data        = 0                              
#########################

# MENGINISIASI DATA TRAINING #
training_cat1 = np.zeros((jumlah_data, jumlah_input)) 
training_dog1 = np.zeros((jumlah_data, jumlah_input))   
training_cat2 = np.zeros((jumlah_data, jumlah_input)) 
training_dog2 = np.zeros((jumlah_data, jumlah_input))   

training1 = np.zeros((jumlah_data, jumlah_output))   
training2 = np.zeros((jumlah_data, jumlah_output))
#########################

training_cat1           = np.load('main/data/{}/{} background/data {}/{}cat1_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2)) 
training_dog1           = np.load('main/data/{}/{} background/data {}/{}dog1_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2))   
training_cat2           = np.load('main/data/{}/{} background/data {}/{}cat2_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2)) 
training_dog2           = np.load('main/data/{}/{} background/data {}/{}dog2_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2))  

mean_training_cat1      = np.mean(training_cat1)
mean_training_dog1      = np.mean(training_dog1)
mean_training_cat2      = np.mean(training_cat2)
mean_training_dog2      = np.mean(training_dog2)

mean1                   = ((mean_training_cat1 + mean_training_dog1) / 2)
mean2                   = ((mean_training_cat2 + mean_training_dog2) / 2)

max1                    = (np.max(training_cat1) + np.max(training_dog1)) / 2
max2                    = (np.max(training_cat2) + np.max(training_dog2)) / 2
min1                    = (np.min(training_cat1) + np.min(training_dog1)) / 2
min2                    = (np.min(training_cat2) + np.min(training_dog2)) / 2

Scaler = MinMaxScaler()
scaled_training_cat1    = Scaler.fit_transform(training_cat1)
scaled_training_dog1    = Scaler.fit_transform(training_dog1)
scaled_training_cat2    = Scaler.fit_transform(training_cat2)
scaled_training_dog2    = Scaler.fit_transform(training_dog2)

if not os.path.exists("main/data/{}/{} background/data {}/training_data_{} sampel".format(keterangan_data, keterangan_background, percobaan_data,panjang_data)): 
    os.mkdir("main/data/{}/{} background/data {}/training_data_{} sampel".format(keterangan_data, keterangan_background, percobaan_data,panjang_data))
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/trainingcat1_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), scaled_training_cat1)
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/trainingcat2_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), scaled_training_cat2)
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/trainingdog1_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), scaled_training_dog1)
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/trainingdog2_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), scaled_training_dog2)

scaled_training_cat1 = scaled_training_cat1.reshape(gambar_data, jumlah_output)
scaled_training_dog1 = scaled_training_dog1.reshape(gambar_data, jumlah_output)
training1[0: gambar_data, 0:jumlah_output] = scaled_training_cat1
training1[gambar_data: jumlah_data, 0:jumlah_output] = scaled_training_dog1
training1 = training1.reshape(jumlah_data * jumlah_output, jumlah_input)
training1 = np.array(training1)

scaled_training_cat2 = scaled_training_cat2.reshape(gambar_data, jumlah_output)
scaled_training_dog2 = scaled_training_dog2.reshape(gambar_data, jumlah_output)
training2[0: gambar_data, 0:jumlah_output] = scaled_training_cat2
training2[gambar_data: jumlah_data, 0:jumlah_output] = scaled_training_dog2
training2 = training2.reshape(jumlah_data * jumlah_output, jumlah_input)
training2 = np.array(training2)

np.save('main/data/{}/{} background/data {}/training_data_{} sampel/mean1_{}d.npy'    .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), mean1)       
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/mean2_{}d.npy'    .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), mean2)       
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/max1_{}d.npy'     .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), max1)        
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/max2_{}d.npy'     .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), max2)        
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/min1_{}d.npy'     .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), min1)        
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/min2_{}d.npy'     .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), min2)        
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/training1_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), training1)   
np.save('main/data/{}/{} background/data {}/training_data_{} sampel/training2_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data), training2)   
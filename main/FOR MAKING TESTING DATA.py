# MEMANGGIL LIBRARY #
import numpy as np
import os, sys
from sklearn.preprocessing import MinMaxScaler
import warnings
#########################

warnings.filterwarnings('ignore')

# HYPER PARAMETER #
jumlah_data = 250
jumlah_output = 2
jumlah_input = 1
jumlah_neuron_hidden = 1000
gambar_data = int(jumlah_data / 2)
nfeature1 = 10000
nfeature2 = 100
panjang_data = jumlah_output * jumlah_data    
keterangan_background = "tanpa"     
percobaan_data      = 0                            
#########################

# MENGINISIASI DATA TESTING #
testing_cat1 = np.zeros((jumlah_data, jumlah_input)) 
testing_dog1 = np.zeros((jumlah_data, jumlah_input))   
testing_cat2 = np.zeros((jumlah_data, jumlah_input)) 
testing_dog2 = np.zeros((jumlah_data, jumlah_input))   

testing1 = np.zeros((jumlah_data, jumlah_output))   
testing2 = np.zeros((jumlah_data, jumlah_output))
#########################

testing_cat1           = np.load('main/data/testing/{} background/data {}/testingcat1_{}d_{}h_{}l.npy'.format(keterangan_background, percobaan_data, jumlah_data, nfeature1, nfeature2)) 
testing_cat2           = np.load('main/data/testing/{} background/data {}/testingcat2_{}d_{}h_{}l.npy'.format(keterangan_background, percobaan_data, jumlah_data, nfeature1, nfeature2))  
testing_dog1           = np.load('main/data/testing/{} background/data {}/testingdog1_{}d_{}h_{}l.npy'.format(keterangan_background, percobaan_data, jumlah_data, nfeature1, nfeature2)) 
testing_dog2           = np.load('main/data/testing/{} background/data {}/testingdog2_{}d_{}h_{}l.npy'.format(keterangan_background, percobaan_data, jumlah_data, nfeature1, nfeature2)) 

mean_testing_cat1      = np.mean(testing_cat1)
mean_testing_dog1      = np.mean(testing_dog1)
mean_testing_cat2      = np.mean(testing_cat2)
mean_testing_dog2      = np.mean(testing_dog2)

mean1                   = ((mean_testing_cat1 + mean_testing_dog1) / 2)
mean2                   = ((mean_testing_cat2 + mean_testing_dog2) / 2)

max1                    = (np.max(testing_cat1) + np.max(testing_dog1)) / 2
max2                    = (np.max(testing_cat2) + np.max(testing_dog2)) / 2
min1                    = (np.min(testing_cat1) + np.min(testing_dog1)) / 2
min2                    = (np.min(testing_cat2) + np.min(testing_dog2)) / 2

Scaler = MinMaxScaler()
scaled_testing_cat1 = Scaler.fit_transform(testing_cat1)
scaled_testing_dog1 = Scaler.fit_transform(testing_dog1)
scaled_testing_cat2 = Scaler.fit_transform(testing_cat2)
scaled_testing_dog2 = Scaler.fit_transform(testing_dog2)

if not os.path.exists("main/data/testing/{} background/data {}/testing_data_{} sampel".format(keterangan_background, percobaan_data, panjang_data)): 
    os.mkdir("main/data/testing/{} background/data {}/testing_data_{} sampel".format(keterangan_background, percobaan_data, panjang_data))
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/testingcat1_{}d.npy'.format(keterangan_background, percobaan_data, panjang_data, jumlah_data), scaled_testing_cat1)
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/testingcat2_{}d.npy'.format(keterangan_background, percobaan_data, panjang_data, jumlah_data), scaled_testing_cat2)
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/testingdog1_{}d.npy'.format(keterangan_background, percobaan_data, panjang_data, jumlah_data), scaled_testing_dog1)
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/testingdog2_{}d.npy'.format(keterangan_background, percobaan_data, panjang_data, jumlah_data), scaled_testing_dog2)

scaled_testing_cat1 = scaled_testing_cat1.reshape(gambar_data, jumlah_output)
scaled_testing_dog1 = scaled_testing_dog1.reshape(gambar_data, jumlah_output)
testing1[0: gambar_data, 0:jumlah_output] = scaled_testing_cat1
testing1[gambar_data: jumlah_data, 0:jumlah_output] = scaled_testing_dog1
testing1 = testing1.reshape(jumlah_data * jumlah_output, jumlah_input)
testing1 = np.array(testing1)

scaled_testing_cat2 = scaled_testing_cat2.reshape(gambar_data, jumlah_output)
scaled_testing_dog2 = scaled_testing_dog2.reshape(gambar_data, jumlah_output)
testing2[0: gambar_data, 0:jumlah_output] = scaled_testing_cat2
testing2[gambar_data: jumlah_data, 0:jumlah_output] = scaled_testing_dog2
testing2 = testing2.reshape(jumlah_data * jumlah_output, jumlah_input)
testing2 = np.array(testing2)

np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/mean1_{}d.npy'       .format(keterangan_background, percobaan_data, panjang_data, panjang_data), mean1)         
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/mean2_{}d.npy'       .format(keterangan_background, percobaan_data, panjang_data, panjang_data), mean2)         
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/max1_{}d.npy'        .format(keterangan_background, percobaan_data, panjang_data, panjang_data), max1)          
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/max2_{}d.npy'        .format(keterangan_background, percobaan_data, panjang_data, panjang_data), max2)          
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/min1_{}d.npy'        .format(keterangan_background, percobaan_data, panjang_data, panjang_data), min1)          
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/min2_{}d.npy'        .format(keterangan_background, percobaan_data, panjang_data, panjang_data), min2)          
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/testing1_{}d.npy'   .format(keterangan_background, percobaan_data, panjang_data, panjang_data), testing1)       
np.save('main/data/testing/{} background/data {}/testing_data_{} sampel/testing2_{}d.npy'   .format(keterangan_background, percobaan_data, panjang_data, panjang_data), testing2)       
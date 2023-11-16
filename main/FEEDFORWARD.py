# MEMANGGIL LIBRARY #
import cv2
import numpy as np
import math 
import glob
import os, sys
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import random
from numpy.random import randn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import serial
import time
#########################

def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def leakyrelu(x, y):
    return np.where(x > 0, x, x * y)

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

# HYPER PARAMETER #
# UNTUK MENENTUKAN NILAI NILAI DARI ALGORITMA FEEDFORWARD #
jumlah_data                         = 250   # 150/300; 200/400; 250/500; 300/600
jumlah_output                       = 2
jumlah_input                        = 1
gambar_data                         = int(jumlah_data / 2)
panjang_data                        = jumlah_data * jumlah_output
number                              = 0
akurasi                             = 0
keterangan_training_background      = "ada"
keterangan_validation_background    = "tanpa"
percobaan_data                      = 0
#########################

# UNTUK NILAI HIGH DAN NILAI LOW DARI ORB #
nfeature1       = 10000
nfeature2       = 100
#########################

# UNTUK CONFUSION MATRIX #
benar           = 0
salah           = 0
label           = 0
#########################

# UNTUK MEMANGGIL DATA HASIL LATIHAN ANN #
hidden          = [10, 20, 40, 80, 160, 320, 640, 1280]
data            = 1000
epochs          = 10000
lr              = ["1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"]
#########################  
#########################

# MEMBUAT FILE PENYIMPAN #
if not os.path.exists("research data/confusion matrix/data {}".format(percobaan_data)): 
    os.mkdir("research data/confusion matrix/data {}".format(percobaan_data))
if not os.path.exists("research data/confusion matrix/data {}/training {} background".format(percobaan_data, keterangan_training_background)): 
    os.mkdir("research data/confusion matrix/data {}/training {} background".format(percobaan_data, keterangan_training_background))
if not os.path.exists("research data/confusion matrix/data {}/training {} background/testing {} background".format(percobaan_data, keterangan_training_background, keterangan_validation_background)): 
    os.mkdir("research data/confusion matrix/data {}/training {} background/testing {} background".format(percobaan_data, keterangan_training_background, keterangan_validation_background))
if not os.path.exists("research data/confusion matrix/data {}/training {} background/testing {} background/{} testing data".format(percobaan_data, keterangan_training_background, keterangan_validation_background, panjang_data)): 
    os.mkdir("research data/confusion matrix/data {}/training {} background/testing {} background/{} testing data".format(percobaan_data, keterangan_training_background, keterangan_validation_background, panjang_data))
#########################

# MENGINISIASI DATA VALIDASI #
validation1     = np.load('main/data/testing/{} background/data {}/testing_data_{} sampel/testing1_{}d.npy'   .format(keterangan_validation_background, percobaan_data, panjang_data, panjang_data))
validation2     = np.load('main/data/testing/{} background/data {}/testing_data_{} sampel/testing2_{}d.npy'   .format(keterangan_validation_background, percobaan_data, panjang_data, panjang_data))  

confusion_m     = np.zeros((jumlah_data, jumlah_output))
LABEL           = np.zeros((jumlah_data, jumlah_output))
predicted       = np.zeros((panjang_data, jumlah_input))
prediksi        = np.zeros((8,8))
#########################
print("JUMLAH DATA:",panjang_data)

LABEL[0: gambar_data, 0:jumlah_output] = 0
LABEL[gambar_data: jumlah_data, 0:jumlah_output] = 1
LABEL = LABEL.reshape(jumlah_data * jumlah_output, jumlah_input)
LABEL = np.array(LABEL)
print(LABEL.shape, predicted.shape)

for i in range(len(hidden)):
    for k in range(len(lr)):
        w1      = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/w1.npy'.format(keterangan_training_background, percobaan_data, hidden[i], str(lr[k]))) # MEMANGGIL WEIGHT PERTAMA
        w2      = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/w2.npy'.format(keterangan_training_background, percobaan_data, hidden[i], str(lr[k]))) # MEMANGGIL WEIGHT KEDUA
        b1      = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/b1.npy'.format(keterangan_training_background, percobaan_data, hidden[i], str(lr[k]))) # MEMANGGIL BIAS PERTAMA
        b2      = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/b2.npy'.format(keterangan_training_background, percobaan_data, hidden[i], str(lr[k]))) # MEMANGGIL BIAS KEDUA      
        for j in range(panjang_data):
            x1 = validation1[j,0]
            x2 = validation2[j,0]
            x_test = np.array([x1, x2])
            # FIRST LAYER
            HL = x_test.T.dot(w1.T) + b1
            # ACTIVATION FUNCTION USING LEAKY RELU #
            AHL = leakyrelu(HL, 0.01)
            ########################################
            # SECOND LAYER
            OL = AHL.dot(w2) + b2
            # ACTIVATION FUNCTION USING SIGMOID #
            AOL = sigmoid(OL)
            #####################################
            # SOFTMAX #
            Y = np.exp(AOL)
            Y1 = Y/ Y.sum()
            ##########
            
            if Y1[1] > Y1[0]:
                label = 1
                predicted[j, 0] = label

            elif Y1[1] < Y1[0]:
                label = 0
                predicted[j, 0] = label

            elif Y1[1] == Y1[0]:
                label = 0
                predicted[j, 0] = label

            if label == LABEL[j, 0]:
                benar = benar + 1
                confusion_m[0, 0] =  benar

            if label != LABEL[j, 0]:
                salah = salah + 1
                confusion_m[0, 1] =  salah
            number += 1
            print("TEBAKAN KE >>>>>>>>>>>> {}".format(number))

        confusion_matrix = metrics.confusion_matrix(LABEL, predicted)

        print("KUCING TEBAK KUCING >>> {}\nKUCING TEBAK ANJING >>> {}\nANJING TEBAK KUCING >>> {}\nANJING TEBAK ANJING >>> {}".format(confusion_matrix[0,0] ,confusion_matrix[0,1] ,confusion_matrix[1,0], confusion_matrix[1,1]))
        prediksi[i , k] = confusion_matrix[0,0] + confusion_matrix[1,1]
        akurasi = (confusion_matrix[0,0] + confusion_matrix[1,1]) / panjang_data
        print("PREDIKSI BENAR >>>>>>>>>>>> {}\nPREDIKSI SALAH >>>>>>>>>>>> {}\nAKURASI >>>>>>>>>>>> {}".format(confusion_matrix[0,0] + confusion_matrix[1,1], confusion_matrix[0,1] + confusion_matrix[1,0], akurasi))

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["KUCING", "ANJING"])
        cm_display.plot()
        plt.savefig('research data/confusion matrix/data {}/training {} background/testing {} background/{} testing data/CM {}N {}D {}E {}H {}L {}LR.png'.format(percobaan_data, keterangan_training_background, keterangan_validation_background, panjang_data, hidden[i], panjang_data, epochs, nfeature1, nfeature2, str(lr[k])))
        plt.close()

prediksi = prediksi.reshape(64,1)
prediksi_pembelajaran = pd.DataFrame(prediksi)
prediksi_pembelajaran.to_excel('research data/prediksi pembelajaraan data {} training {} background testing {} background.xlsx'.format(percobaan_data, keterangan_training_background, keterangan_validation_background), index = False)
# MEMANGGIL LIBRARY #
import cv2
import numpy as np
import math as m
import pandas as pd
import os, sys
import warnings
import time
from matplotlib import pyplot as plt
from numpy.random import randn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
#########################

warnings.filterwarnings('ignore')

def leakyrelu(x, y):
    return np.where(x > 0, x, x * y)

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def der_leakyrelu(x, y):
    return np.where(x > 0, 1, y)

# HYPER PARAMETER #
jumlah_data             = 1000
jumlah_output           = 2
jumlah_input            = 1
jumlah_neuron_hidden    = [10, 20, 40, 80, 160, 320, 640, 1280]
gambar_data             = int(jumlah_data / 2)
nfeature1               = 10000 # 10000
nfeature2               = 100   # 100/250/500
no                      = 0
lr                      = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
lr1                     = ["1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"]
epochs                  = 10000
panjang_data            = jumlah_output * jumlah_data   
rmse                    = 1     
keterangan_background   = "tanpa" 
keterangan_data         = "training"   
percobaan_data          = 0                   
#########################

# PEMBUATAN DATA INPUT UNTUK PERCOBAAN ANN #
pca_satu    = PCA(n_components = 1, svd_solver='auto')
pca_dua     = PCA(n_components = 1, svd_solver='auto')
scaler_satu = StandardScaler()
scaler_dua  = StandardScaler()
orbsatu     = cv2.ORB_create(nfeatures=nfeature1)
orbdua      = cv2.ORB_create(nfeatures=nfeature2)

image       = cv2.imread("PROGRAM/POLIKISTIK-RENALIS-PADA-ANJING-GOLDEN-RETRIEVER-600x353.jpg")
#image      = cv2.imread("PROGRAM/kucing_11zon.jpg")

resized_image   = cv2.resize(image, (200, 100),  interpolation = cv2.INTER_AREA) 
gray_image      = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
kpsatu, dessatu = orbsatu.detectAndCompute(gray_image, None)
kpdua, desdua   = orbdua.detectAndCompute(gray_image, None)

if dessatu is not None and desdua is not None:
    descriptors_normalized_satu = scaler_satu.fit_transform(dessatu)
    descriptors_normalized_dua  = scaler_dua.fit_transform(desdua)
    principal_components_satu   = pca_satu.fit_transform(descriptors_normalized_satu)
    principal_components_dua    = pca_dua.fit_transform(descriptors_normalized_dua)

high  = np.mean(principal_components_satu)
low   = np.mean(principal_components_dua)
#########################

# MENGINISIASI DATA TRAINING DAN DATA VALIDASI #
validation_cat1         = np.zeros((jumlah_data, jumlah_input))  
validation_dog1         = np.zeros((jumlah_data, jumlah_input))  
validation_cat2         = np.zeros((jumlah_data, jumlah_input))  
validation_dog2         = np.zeros((jumlah_data, jumlah_input))  

training1               = np.zeros((jumlah_data, jumlah_output))   
validation1             = np.zeros((jumlah_data, jumlah_output), dtype='S')
training2               = np.zeros((jumlah_data, jumlah_output))
validation2             = np.zeros((jumlah_data, jumlah_output), dtype='S')

OUTPUT                  = np.zeros((panjang_data, jumlah_output*jumlah_output))
#########################

# VARIBEL ANN YANG DIGUNAKAN #	                                               
mean1                   = np.load('main/data/{}/{} background/data {}/training_data_{} sampel/mean1_{}d.npy'    .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data))
mean2                   = np.load('main/data/{}/{} background/data {}/training_data_{} sampel/mean2_{}d.npy'    .format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data))

Z                       = np.zeros((panjang_data, jumlah_output))    
U                       = np.zeros((jumlah_output, panjang_data))    
convergence             = np.zeros((epochs, jumlah_input))  
durasi                  = np.zeros((8, 8))  
#########################

print("JUMLAH DATA:",panjang_data)

training1                                               = np.load('main/data/{}/{} background/data {}/training_data_{} sampel/training1_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data))
training2                                               = np.load('main/data/{}/{} background/data {}/training_data_{} sampel/training2_{}d.npy'.format(keterangan_data, keterangan_background, percobaan_data,panjang_data, panjang_data))

validation1[0: gambar_data, 0:jumlah_output]            = 'A'
validation1[gambar_data: jumlah_data, 0:jumlah_output]  = 'B'
validation1                                             = validation1.reshape(panjang_data, jumlah_input)
validation1                                             = np.array(validation1)
onehot                                                  = OneHotEncoder(handle_unknown='ignore')
onehot_ds                                               = pd.DataFrame(onehot.fit_transform(validation1).toarray())

OUTPUT[0:, 0:1]                                         = training1
OUTPUT[0:, 1:2]                                         = training2
OUTPUT[0:, 2:4]                                         = onehot_ds
print(OUTPUT)

# PEMBUATAN DATA INPUT #
X                                           = np.hstack([(training1), (training2)])
Y                                           = np.hstack([validation1[0:panjang_data, 0:jumlah_input]])
dataset                                     = pd.DataFrame(X, columns={"X1", "X2"})
dataset["Y"]                                = Y
enc                                         = OneHotEncoder(handle_unknown='ignore')
enc_ds                                      = pd.DataFrame(enc.fit_transform(dataset[["Y"]]).toarray())
final_ds                                    = dataset.join(enc_ds)
final_ds.drop("Y", axis=1, inplace=True)    
final_ds.columns                            = ['X2', 'X1','Kucing', 'Anjing']
OP                                          = pd.DataFrame(OUTPUT)
OP                                          = OP.sample(frac=1)
OP.columns                                  = ['x1', 'x2','y1', 'y2']
i1                                          = np.array(OP.get('x1'))
i2                                          = np.array(OP.get('x2'))
y1                                          = np.array(OP.get('y1'))
y2                                          = np.array(OP.get('y2'))
U[0, :]                                     = i1  
U[1, :]                                     = i2
Z[:, 0]                                     = y1                                                             
Z[:, 1]                                     = y2
Z1                                          = Z[:, 1]
#########################

def forward_prop(X, wi_1, bi_1, wi_2, bi_2):
    # HIDDEN LAYER
    HL = X.T.dot(wi_1.T) + bi_1 
    # ACTIVATION FUNCTION USING LEAKY RELU #
    AHL = leakyrelu(HL, 0.01) 
    #########################################
    # OUTPUT LAYER
    OL = AHL.dot(wi_2) + bi_2
    # ACTIVATION FUNCTION USING SIGMOID #
    AOL = sigmoid(OL)  
    #####################################
    # SOFTMAX #
    H = np.exp(AOL)
    Y = H / H.sum(axis=1, keepdims=True)
    ###########
    return Y, AHL, HL  

def diff_wi_2(H, Z, Y):                                                                    
    return H.T.dot(Z - Y)  # TURUNAN UNTUK WEIGHT KEDUA 

def diff_wi_1(X, H, Z, output, wi_2):                                                       
    dZ = (Z - output).dot(wi_2.T) * der_leakyrelu(H, 0.01)
    return X.dot(dZ)  # TURURAN UNTUK WEIGHT PERTAMA

def diff_bi_2(Z, Y):                                                                        
    return (Z - Y).sum(axis=0)  # TURURAN UNTUK BIAS KEDUA

def diff_bi_1(Z, Y, wi_2, H):                                                               
    return ((Z - Y).dot(wi_2.T) * der_leakyrelu(H, 0.01)).sum(axis=0)  # TURURAN UNTUK BIAS PERTAMA

for i in range(len(jumlah_neuron_hidden)):
    for j in range(len(lr)):
        start = time.time()
        w1 = randn(jumlah_neuron_hidden[i], jumlah_output)        	                                            
        w2 = randn(jumlah_neuron_hidden[i], jumlah_output)       	                                                
        b1 = randn(jumlah_neuron_hidden[i])          	                                               
        b2 = randn(jumlah_output)  
        for epoch in range(epochs):
            output, act_hidden_layer, hidden_layer = forward_prop(U, w1, b1, w2, b2)
            w2 += lr[j] * diff_wi_2(act_hidden_layer, Z, output)
            b2 += lr[j] * diff_bi_2(Z, output)
            w1 += lr[j] * diff_wi_1(U, hidden_layer, Z, output, w2).T
            b1 += lr[j] * diff_bi_1(Z, output, w2, hidden_layer)        	                                            
            w2 = np.nan_to_num(w2)
            b2 = np.nan_to_num(b2)
            w1 = np.nan_to_num(w1)      	                                                
            b1 = np.nan_to_num(b1)        	                                               
            mse = np.mean(np.square(np.subtract(output[:,0], Z[:,0])))
            rmse = m.sqrt(mse)
            convergence[epoch,0] = rmse
            no += 1
            print("BELAJAR KE >>> {} RMSE >>> {}".format(no, rmse))
            if no == epochs:
                no = 0
                break

        x1 = high / mean1
        x2 = low / mean2

        print(x1, x2)

        x_test = np.array([x1, x2])
        print(x_test.shape)

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
        print("prob of class CATTO >>>>>>>>>>>> {}\nprob of class DOGGO >>>>>>>>>>>> {}".format(Y1[0], Y1[1]))

        # MEMBUAT FILE PENYIMPAN #
        if not os.path.exists("main/input data weight dan bias/{} background/data {}".format(keterangan_background, percobaan_data)): 
            os.mkdir("main/input data weight dan bias/{} background/data {}".format(keterangan_background, percobaan_data))
        if not os.path.exists("main/input data weight dan bias/{} background/data {}/{} neuron".format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i])): 
            os.mkdir("main/input data weight dan bias/{} background/data {}/{} neuron".format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i]))
        if not os.path.exists("main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate".format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i], str(lr1[j]))): 
            os.mkdir("main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate".format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i], str(lr1[j])))

        if not os.path.exists("research data/RMSE/data {}".format(percobaan_data)): 
            os.mkdir("research data/RMSE/data {}".format(percobaan_data))
        if not os.path.exists("research data/RMSE/data {}/training {} background".format(percobaan_data, keterangan_background)): 
            os.mkdir("research data/RMSE/data {}/training {} background".format(percobaan_data, keterangan_background))
        #########################
        
        # MENYIMPAN DATA WEIGHT1, WEIGHT2, BIAS1, DAN BIAS2 #
        np.save('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/w1.npy'.format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i], str(lr1[j])), w1)           # MENYIMPAN WEIGHT PERTAMA
        np.save('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/w2.npy'.format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i], str(lr1[j])), w2)           # MENYIMPAN WEIGHT KEDUA
        np.save('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/b1.npy'.format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i], str(lr1[j])), b1)           # MENYIMPAN BIAS PERTAMA
        np.save('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/b2.npy'.format(keterangan_background, percobaan_data, jumlah_neuron_hidden[i], str(lr1[j])), b2)           # MENYIMPAN BIAS KEDUA
        #########################

        plt.plot(convergence, '.r',label ='Inline label')
        plt.legend(["RMSE"])
        plt.title("RMSE {}n {}d {}e {}h {}l {}lr".format(jumlah_neuron_hidden[i], jumlah_data, epochs, nfeature1, nfeature2, lr1[j]))
        plt.savefig('research data/RMSE/data {}/training {} background/RMSE {}N {}E {}H {}L {}LR.png'.format(percobaan_data, keterangan_background, jumlah_neuron_hidden[i], epochs, nfeature1, nfeature2, str(lr1[j]))) 
        plt.close()

        # DURASI KOMPUTASI #
        end = time.time()
        duration = end - start
        print("DURATION >>>>>>>>>>>> {}".format(duration))
        durasi[i, j] = duration
        duration = 0
        #########################

durasi = durasi.reshape(64, 1)
durasi_pembelajaran = pd.DataFrame(durasi)
durasi_pembelajaran.to_excel('research data/learning duration data {} training {} background.xlsx'.format(percobaan_data, keterangan_background), index = False)

cv2.waitKey(0)
cv2.destroyAllWindows()
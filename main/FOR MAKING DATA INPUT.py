# MEMANGGIL LIBRARY #
import cv2
import numpy as np
import glob
import os, sys
from matplotlib import pyplot as plt
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time                              
#########################

start = time.time()

warnings.filterwarnings('ignore')

# HYPER PARAMETER #
jumlah_data             = 250
jumlah_output           = 2
jumlah_input_data       = 2
jumlah_input            = 1
jumlah_neuron_hidden    = 1000
gambar_data             = int(jumlah_data / 2)
nfeature1               = 10000
nfeature2               = 250          # 100/0; 250/1; 500/2
epochs                  = 10000
panjang_data            = jumlah_output * jumlah_data  
keterangan_data         = "testing"    # training OR testing
keterangan_background   = "ada"        # ada ATAU tanpa    
format_gambar           = "jpg"        # jpg for DATA TESTING; png for DATA TRAINING  
percobaan_data          = 0            # 0/100; 1/250; 2/500  
data_pca                = 100                                 
#########################

# MEMANGGIL GAMBAR #
pic_cat = "data/{}/cats {} background/*.{}*".format(keterangan_data, keterangan_background, format_gambar)
pic_dog = "data/{}/dogs {} background/*.{}*".format(keterangan_data, keterangan_background, format_gambar)

if not os.path.exists("main/data"): 
    os.mkdir("main/data")
if not os.path.exists("main/data/{}".format(keterangan_data)): 
    os.mkdir("main/data/{}".format(keterangan_data))
if not os.path.exists("main/data/{}/{} background".format(keterangan_data, keterangan_background)): 
    os.mkdir("main/data/{}/{} background".format(keterangan_data, keterangan_background))
if not os.path.exists("main/data/{}/{} background/data {}".format(keterangan_data, keterangan_background, percobaan_data)): 
    os.mkdir("main/data/{}/{} background/data {}".format(keterangan_data, keterangan_background, percobaan_data))
#########################

# MENGINISIASI DATA TRAINING DAN DATA VALIDASI #
cat1 = np.zeros((jumlah_data, jumlah_input)) 
cat2 = np.zeros((jumlah_data, jumlah_input)) 
dog1 = np.zeros((jumlah_data, jumlah_input))   
dog2 = np.zeros((jumlah_data, jumlah_input))     
#########################

# V A R I B E L  O R B  P C A  Y A N G  D I G U N A K A N #
pca11 = PCA(n_components = 1, svd_solver = 'full')
pca21 = PCA(n_components = 1, svd_solver = 'full')
pca12 = PCA(n_components = 1, svd_solver = 'full')
pca22 = PCA(n_components = 1, svd_solver = 'full')

orb11 = cv2.ORB_create(nfeatures = nfeature1)
orb21 = cv2.ORB_create(nfeatures = nfeature1)
orb12 = cv2.ORB_create(nfeatures = nfeature2)
orb22 = cv2.ORB_create(nfeatures = nfeature2)

scaler11 = StandardScaler()
scaler21 = StandardScaler()
scaler12 = StandardScaler()
scaler22 = StandardScaler()
#########################

print("jumlah data: {}   data: {}   background: {} background".format(panjang_data, keterangan_data, keterangan_background))

def gambar(jumlah_data, nfeature1, nfeature2):
    img_cat = 0
    img_dog = 0

    for imgs1 in glob.glob(pic_cat):
        for i in range(len(cat1) - 1, 0, -1):
            img11 = cv2.imread(imgs1)
            dim_1 = (200, 100)
            img12 = cv2.resize(img11, dim_1,  interpolation = cv2.INTER_AREA) 
            img13 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
            kp11, des11 = orb11.detectAndCompute(img13, None)
            kp12, des12 = orb12.detectAndCompute(img13, None)
            if des11 is not None and des12 is not None:
                descriptors_normalized11    = scaler11.fit_transform(des11.reshape(-1, 1))
                descriptors_normalized12    = scaler12.fit_transform(des12.reshape(-1, 1))
                principal_components11      = pca11.fit_transform(descriptors_normalized11)
                principal_components12      = pca12.fit_transform(descriptors_normalized12)
                cat1[i] = cat1[i - 1]
                cat2[i] = cat2[i - 1]
            else:
                continue
        cat1[0] = np.mean(principal_components11[:data_pca])
        cat2[0] = np.mean(principal_components12[:data_pca])
        print("cat1 >>>>>>>>>>>> {}\ncat2 >>>>>>>>>>>> {}".format(cat1[0], cat2[0]))
        print("GAMBAR DATA KUCING KE:", img_cat + 1)
        img_cat += 1
        if img_cat == jumlah_data:
            break

    for imgs2 in glob.glob(pic_dog):
        for i in range(len(dog1) - 1, 0, -1):
            img21 = cv2.imread(imgs2)   
            dim_2 = (200, 100)      
            img22 = cv2.resize(img21, dim_2,  interpolation = cv2.INTER_AREA) 
            img23 = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
            kp21, des21 = orb21.detectAndCompute(img23, None)
            kp22, des22 = orb22.detectAndCompute(img23, None)
            if des21 is not None and des22 is not None:
                descriptors_normalized21    = scaler21.fit_transform(des21.reshape(-1, 1))
                descriptors_normalized22    = scaler22.fit_transform(des22.reshape(-1, 1))
                principal_components21      = pca21.fit_transform(descriptors_normalized21)
                principal_components22      = pca22.fit_transform(descriptors_normalized22)
                dog1[i] = dog1[i - 1]
                dog2[i] = dog2[i - 1]
            else:
                continue
        dog1[0] = np.mean(principal_components21[:data_pca])
        dog2[0] = np.mean(principal_components22[:data_pca])
        print("dog1 >>>>>>>>>>>> {}\ndog2 >>>>>>>>>>>> {}".format(dog1[0], dog2[0]))
        print("GAMBAR DATA ANJING KE:", img_dog + 1)
        img_dog += 1
        if img_dog == jumlah_data:
            break

    return cat1, dog1, cat2, dog2

gambar(jumlah_data, nfeature1, nfeature2)
    
np.save('main/data/{}/{} background/data {}/{}cat1_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2), cat1) 
np.save('main/data/{}/{} background/data {}/{}cat2_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2), cat2)   
np.save('main/data/{}/{} background/data {}/{}dog1_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2), dog1) 
np.save('main/data/{}/{} background/data {}/{}dog2_{}d_{}h_{}l.npy'.format(keterangan_data, keterangan_background, percobaan_data, keterangan_data, jumlah_data, nfeature1, nfeature2), dog2) 

# DURASI KOMPUTASI #
end = time.time()
duration = end - start
duration = str(duration).replace('.', ',')
print("DURATION >>>>>>>>>>>> {}".format(duration))
####################################################
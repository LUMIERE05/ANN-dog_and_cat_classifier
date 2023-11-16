# MEMANGGIL LIBRARY #
import cv2
import numpy as np
import tkinter as tk
import math 
import os, sys
import serial
import time
import threading
import warnings
from serial.serialutil import SerialTimeoutException
from statistics import mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threading import Thread
from PIL import Image, ImageTk
from turtle import *
#########################

warnings.filterwarnings('ignore')

# HYPER PARAMETER #
# UNTUK MENENTUKAN NILAI NILAI DARI ALGORITMA FEEDFORWARD #
jumlah_output           = 2
jumlah_input            = 1
jumlah_data_testing     = 500
#########################

# UNTUK NILAI HIGH DAN NILAI LOW DARI ORB #
orb_nfeature_high       = 10000
orb_nfeature_low        = 100
#########################

# UNTUK MEMANGGIL DATA HASIL LATIHAN ANN #
hidden                  = 20
data                    = 1000
epochs                  = 10000
lr                      = "1e-03"
#########################  

# PARAMETER KAMERA #
dimensi_gambar          = (200, 100)
#########################
# KETERANGAN GAMBAR
keterangan_gambar_ff    = "testing"
keterangan_training     = "tanpa"
keterangan_testing      = "tanpa"
data_gambar             = 0
#########################

# MEMANGGIL DATA HASIL LATIHAN ANN #
w1  = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/w1.npy'.format(keterangan_training, data_gambar, hidden, str(lr))) # MEMANGGIL WEIGHT PERTAMA
w2  = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/w2.npy'.format(keterangan_training, data_gambar, hidden, str(lr))) # MEMANGGIL WEIGHT KEDUA
b1  = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/b1.npy'.format(keterangan_training, data_gambar, hidden, str(lr))) # MEMANGGIL BIAS PERTAMA
b2  = np.load('main/input data weight dan bias/{} background/data {}/{} neuron/{} learning rate/b2.npy'.format(keterangan_training, data_gambar, hidden, str(lr))) # MEMANGGIL BIAS KEDUA
#########################

# MEMANGGIL DATA VALIDASI #
max1        = np.load('main/data/{}/{} background/data {}/{}_data_{} sampel/max1_{}d.npy'.format(keterangan_gambar_ff, keterangan_testing, data_gambar, keterangan_gambar_ff, jumlah_data_testing, jumlah_data_testing)) # MEMANGGIL DATA MAXIMUM UNTUK ORB HIGH DARI DATA TESTING
max2        = np.load('main/data/{}/{} background/data {}/{}_data_{} sampel/max2_{}d.npy'.format(keterangan_gambar_ff, keterangan_testing, data_gambar, keterangan_gambar_ff, jumlah_data_testing, jumlah_data_testing)) # MEMANGGIL DATA MAXIMUM UNTUK ORB LOW DARI DATA TESTING
min1        = np.load('main/data/{}/{} background/data {}/{}_data_{} sampel/min1_{}d.npy'.format(keterangan_gambar_ff, keterangan_testing, data_gambar, keterangan_gambar_ff, jumlah_data_testing, jumlah_data_testing)) # MEMANGGIL DATA MINIMUM UNTUK ORB HIGH DARI DATA TESTING
min2        = np.load('main/data/{}/{} background/data {}/{}_data_{} sampel/min2_{}d.npy'.format(keterangan_gambar_ff, keterangan_testing, data_gambar, keterangan_gambar_ff, jumlah_data_testing, jumlah_data_testing)) # MEMANGGIL DATA MINIMUM UNTUK ORB LOW DARI DATA TESTING
mean1       = np.load('main/data/{}/{} background/data {}/{}_data_{} sampel/min1_{}d.npy'.format(keterangan_gambar_ff, keterangan_testing, data_gambar, keterangan_gambar_ff, jumlah_data_testing, jumlah_data_testing)) # MEAMNGGIL DATA MEAN VALIDASI HIGH 
mean2       = np.load('main/data/{}/{} background/data {}/{}_data_{} sampel/min2_{}d.npy'.format(keterangan_gambar_ff, keterangan_testing, data_gambar, keterangan_gambar_ff, jumlah_data_testing, jumlah_data_testing)) # MEAMNGGIL DATA MEAN VALIDASI LOW
#########################

# INISIASI VARIABEL YANG DIPERLUKAN #
MULAI                 = 1
periode               = 3
#########################

# MENYIMPAN HASIL AKHIR ORB PCA ANN #
HASILAKHIR_GAMBAR1_KUCING = np.zeros((jumlah_input, jumlah_input))
HASILAKHIR_GAMBAR1_ANJING = np.zeros((jumlah_input, jumlah_input))
HASILAKHIR_GAMBAR2_KUCING = np.zeros((jumlah_input, jumlah_input))
HASILAKHIR_GAMBAR2_ANJING = np.zeros((jumlah_input, jumlah_input))
#########################

# M E N Y I M P A N   D A T A   H A S I L   A K H I R #
subset_data_front_kucing_gambar1 = []
subset_data_front_anjing_gambar1 = []
subset_data_front_kucing_gambar2 = []
subset_data_front_anjing_gambar2 = []
data_front_kucing_gambar1   = []
data_front_anjing_gambar1   = []
data_front_kucing_gambar2   = []
data_front_anjing_gambar2   = []
kucing_gambar1   = []
anjing_gambar1   = []
kucing_gambar2   = []
anjing_gambar2   = []
#########################

kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
kamera.set(cv2.CAP_PROP_FOCUS, 255)

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

def pengolahan_data_kamera_depan():
    principal_components_front_gambar1_high = 0
    principal_components_front_gambar1_low  = 0
    principal_components_front_gambar2_high = 0
    principal_components_front_gambar2_low  = 0  
    
    while True:
        ret_front1, frame_front1 = kamera.read()
        ret_front2, frame_front2 = kamera.read()

        frame_front_gambar1 = frame_front1[220:450, 380:650]
        frame_front_gambar2 = frame_front2[220:450, 830:1210]

        # ORB PCA GAMBAR 1 #
        resized_front_gambar1     = cv2.resize(frame_front_gambar1, dimensi_gambar,  interpolation = cv2.INTER_AREA) 
        converted_front_gambar1   = cv2.cvtColor(resized_front_gambar1, cv2.COLOR_BGR2GRAY)
        converted_front_gambar1   = cv2.equalizeHist(converted_front_gambar1)
        orb_gambar1_high    = cv2.ORB_create(nfeatures=orb_nfeature_high)
        orb_gambar1_low     = cv2.ORB_create(nfeatures=orb_nfeature_low)
        keypoints_front_gambar1_high, descriptor_front_gambar1_high = orb_gambar1_high.detectAndCompute(converted_front_gambar1, None)
        keypoints_front_gambar1_low, descriptor_front_gambar1_low   = orb_gambar1_low.detectAndCompute(converted_front_gambar1, None)
        scaler_gambar1_high = StandardScaler()
        scaler_gambar1_low  = StandardScaler()
        pca_gambar1_high    = PCA(n_components = 1, svd_solver='auto')
        pca_gambar1_low     = PCA(n_components = 1, svd_solver='auto')
        #########################

        # ORB PCA GAMBAR 2 #
        resized_front_gambar2     = cv2.resize(frame_front_gambar2, dimensi_gambar,  interpolation = cv2.INTER_AREA) 
        converted_front_gambar2   = cv2.cvtColor(resized_front_gambar2, cv2.COLOR_BGR2GRAY)
        converted_front_gambar2   = cv2.equalizeHist(converted_front_gambar2)
        orb_gambar2_high    = cv2.ORB_create(nfeatures=orb_nfeature_high)
        orb_gambar2_low     = cv2.ORB_create(nfeatures=orb_nfeature_low)
        keypoints_front_gambar2_high, descriptor_front_gambar2_high = orb_gambar2_high.detectAndCompute(converted_front_gambar2, None)
        keypoints_front_gambar2_low, descriptor_front_gambar2_low   = orb_gambar2_low.detectAndCompute(converted_front_gambar2, None)
        scaler_gambar2_high = StandardScaler()
        scaler_gambar2_low  = StandardScaler()
        pca_gambar2_high    = PCA(n_components = 1, svd_solver='auto')
        pca_gambar2_low     = PCA(n_components = 1, svd_solver='auto')
        #########################

        if descriptor_front_gambar1_low is not None and descriptor_front_gambar1_high is not None:
            descriptors_normalized_front_gambar1_high = scaler_gambar1_high.fit_transform(descriptor_front_gambar1_high.reshape(-1, 1))
            descriptors_normalized_front_gambar1_low  = scaler_gambar1_low.fit_transform(descriptor_front_gambar1_low.reshape(-1, 1))
            principal_components_front_gambar1_high   = pca_gambar1_high.fit_transform(descriptors_normalized_front_gambar1_high)
            principal_components_front_gambar1_low    = pca_gambar1_low.fit_transform(descriptors_normalized_front_gambar1_low)

        if descriptor_front_gambar2_low is not None and descriptor_front_gambar2_high is not None:
            descriptors_normalized_front_gambar2_high = scaler_gambar2_high.fit_transform(descriptor_front_gambar2_high.reshape(-1, 1))
            descriptors_normalized_front_gambar2_low  = scaler_gambar2_low.fit_transform(descriptor_front_gambar2_low.reshape(-1, 1))
            principal_components_front_gambar2_high   = pca_gambar2_high.fit_transform(descriptors_normalized_front_gambar2_high)
            principal_components_front_gambar2_low    = pca_gambar2_low.fit_transform(descriptors_normalized_front_gambar2_low)

        else:
            continue

        testing_front_gambar1_high    = np.mean(principal_components_front_gambar1_high[:100])
        testing_front_gambar1_low     = np.mean (principal_components_front_gambar1_low[:100])
        testing_front_gambar2_high    = np.mean(principal_components_front_gambar2_high[:100])
        testing_front_gambar2_low     = np.mean (principal_components_front_gambar2_low[:100])

        normalized_testing_front_gambar1_high     = (testing_front_gambar1_high - min1) / (max1 - min1)
        normalized_testing_front_gambar1_low      = (testing_front_gambar1_low  - min2) / (max2 - min2) 
        normalized_testing_front_gambar2_high     = (testing_front_gambar2_high - min1) / (max1 - min1)
        normalized_testing_front_gambar2_low      = (testing_front_gambar2_low  - min2) / (max2 - min2) 
        #########################
        
        if normalized_testing_front_gambar2_high is None or normalized_testing_front_gambar2_high is None or normalized_testing_front_gambar2_high is None or normalized_testing_front_gambar2_high is None:
            print("TIDAK ADA APA-APA")

        else:
            # ANN GAMBAR 1 DAN GAMBAR 2 #
            x1_ann_gambar1_high = normalized_testing_front_gambar1_high
            x2_ann_gambar1_low  = normalized_testing_front_gambar1_low
            x1_ann_gambar2_high = normalized_testing_front_gambar2_high
            x2_ann_gambar2_low  = normalized_testing_front_gambar2_low

            x_test_gambar1      = np.array([x1_ann_gambar1_high, x2_ann_gambar1_low])
            x_test_gambar2      = np.array([x1_ann_gambar2_high, x2_ann_gambar2_low])
            
            # FIRST LAYER
            HL_GAMBAR1 = x_test_gambar1.T.dot(w1.T) + b1
            HL_GAMBAR2 = x_test_gambar2.T.dot(w1.T) + b1
            # ACTIVATION FUNCTION USING LEAKY RELU #
            AHL_GAMBAR1 = leakyrelu(HL_GAMBAR1, 0.01)
            AHL_GAMBAR2 = leakyrelu(HL_GAMBAR2, 0.01)
            #########################
            # SECOND LAYER
            OL_GAMBAR1 = AHL_GAMBAR1.dot(w2) + b2
            OL_GAMBAR2 = AHL_GAMBAR2.dot(w2) + b2
            # ACTIVATION FUNCTION USING SIGMOID #
            AOL_GAMBAR1 = sigmoid(OL_GAMBAR1)
            AOL_GAMBAR2 = sigmoid(OL_GAMBAR2)
            #########################
            # SOFTMAX #
            Y_GAMBAR1 = np.exp(AOL_GAMBAR1)
            Y_GAMBAR2 = np.exp(AOL_GAMBAR2)
            HASILAKHIR_GAMBAR1 = Y_GAMBAR1/ Y_GAMBAR1.sum()
            HASILAKHIR_GAMBAR2 = Y_GAMBAR2/ Y_GAMBAR2.sum()
            #########################
            # HASIL AKHIR #
            HASILAKHIR_GAMBAR1_KUCING = HASILAKHIR_GAMBAR1[0]
            HASILAKHIR_GAMBAR1_ANJING = HASILAKHIR_GAMBAR1[1]
            HASILAKHIR_GAMBAR2_KUCING = HASILAKHIR_GAMBAR2[0]
            HASILAKHIR_GAMBAR2_ANJING = HASILAKHIR_GAMBAR2[1]
            #########################
            #########################
            
            kucing1 = truncate(HASILAKHIR_GAMBAR1_KUCING, 3)
            anjing1 = truncate(HASILAKHIR_GAMBAR1_ANJING, 3)
            kucing2 = truncate(HASILAKHIR_GAMBAR2_KUCING, 3)
            anjing2 = truncate(HASILAKHIR_GAMBAR2_ANJING, 3)

            data_front_kucing_gambar1.append(kucing1)
            data_front_anjing_gambar1.append(anjing1)
            data_front_kucing_gambar2.append(kucing2)
            data_front_anjing_gambar2.append(anjing2)
            # print (data_front_kucing_gambar1)
            if len(data_front_kucing_gambar1) >= periode and len(data_front_anjing_gambar1) >= periode and len(data_front_kucing_gambar2) >= periode and len(data_front_anjing_gambar2) >= periode:
                
                subset_data_front_kucing_gambar1 = data_front_kucing_gambar1[-periode:]
                subset_data_front_anjing_gambar1 = data_front_anjing_gambar1[-periode:]
                subset_data_front_kucing_gambar2 = data_front_kucing_gambar2[-periode:]
                subset_data_front_anjing_gambar2 = data_front_anjing_gambar2[-periode:]

                unique_kucing_gambar1, count_kucing_gambar1 = np.unique(subset_data_front_kucing_gambar1, return_counts = True)
                unique_anjing_gambar1, count_anjing_gambar1 = np.unique(subset_data_front_anjing_gambar1, return_counts = True)
                unique_kucing_gambar2, count_kucing_gambar2 = np.unique(subset_data_front_kucing_gambar2, return_counts = True)
                unique_anjing_gambar2, count_anjing_gambar2 = np.unique(subset_data_front_anjing_gambar2, return_counts = True)

                max_index_kucing_gambar1 = np.argmax(count_kucing_gambar1)
                max_index_anjing_gambar1 = np.argmax(count_anjing_gambar1)
                max_index_kucing_gambar2 = np.argmax(count_kucing_gambar2)
                max_index_anjing_gambar2 = np.argmax(count_anjing_gambar2)

                modus_kucing_gambar1 = unique_kucing_gambar1[max_index_kucing_gambar1]
                modus_anjing_gambar1 = unique_anjing_gambar1[max_index_anjing_gambar1]
                modus_kucing_gambar2 = unique_kucing_gambar2[max_index_kucing_gambar2]
                modus_anjing_gambar2 = unique_anjing_gambar2[max_index_anjing_gambar2]

                kucing_gambar1.append(modus_kucing_gambar1)
                anjing_gambar1.append(modus_anjing_gambar1)
                kucing_gambar2.append(modus_kucing_gambar2)
                anjing_gambar2.append(modus_anjing_gambar2)

                if len(data_front_kucing_gambar1) == periode + 1 and len(data_front_anjing_gambar1) == periode + 1 and len(data_front_kucing_gambar2) == periode + 1 and len(data_front_anjing_gambar2) == periode + 1:
                    data_front_kucing_gambar1.clear()
                    data_front_anjing_gambar1.clear()
                    data_front_kucing_gambar2.clear()
                    data_front_anjing_gambar2.clear()   

                else:
                    pass     
                
            else:
                pass


def kirim_data():
    kucing1_arduino = []
    anjing1_arduino = []
    kucing2_arduino = []
    anjing2_arduino = []
    s = 1
    Arduino = serial.Serial('COM1', 115200, timeout = 5)
    retry_count = 0

    while s == 1:
        # Menghasilkan dua angka acak untuk frame pertama
        number1_left1 = kucing_gambar1
        number1_right1 = anjing_gambar1

        # Menghasilkan dua angka acak untuk frame kedua
        number2_left1 =  kucing_gambar2
        number2_right1 =  anjing_gambar2

        if len(number1_left1) > 0 and len(number1_right1) > 0 and len(number2_left1) > 0 and len(number2_right1) > 0:
            kucing1_arduino = number1_left1  [-1]
            anjing1_arduino = number1_right1 [-1]
            kucing2_arduino = number2_left1  [-1]
            anjing2_arduino = number2_right1 [-1]

            if len(number1_left1) > 3 and len(number1_right1) > 3 and len(number2_left1) > 3 and len(number2_right1) > 3:
                number1_left1   .pop(0)
                number1_right1  .pop(0)
                number2_left1   .pop(0)
                number2_right1  .pop(0)

            else:
                pass
        
            print(number1_left1)
        else:
            kucing1_arduino = 0.5
            anjing1_arduino = 0.5
            kucing2_arduino = 0.5
            anjing2_arduino = 0.5
        
        # print(kucing_arduino,anjing_arduino, kucing_arduino, anjing_arduino)

        try:
            Arduino.write((str(kucing1_arduino) + 'a' + str(anjing1_arduino) + 'b' + str(kucing2_arduino) + 'c' + str(anjing2_arduino) + 'd').encode('utf-8'))
       
        except SerialTimeoutException:
            retry_count += 1
            print("error!!!", "retry: ", retry_count)

def main():
    kucing1 = []
    anjing1 = []
    kucing2 = []
    anjing2 = []

    pilihan_1 = []
    pilihan_2 = []

    ret1, frame1 = kamera.read()
    ret2, frame2 = kamera.read()
    frame1 = frame1[240:450, 300:650]
    frame2 = frame2[240:450, 850:1210]

    if ret1 and ret2:
        # Mengubah format warna frame dari BGR ke RGB
        rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Menghasilkan dua angka acak untuk frame pertama
        number1_left = kucing_gambar1
        number1_right = anjing_gambar1

        # Menghasilkan dua angka acak untuk frame kedua
        number2_left =  kucing_gambar2
        number2_right =  anjing_gambar2

        if len(number1_left) > 0 and len(number1_right) > 0 and len(number2_left) > 0 and len(number2_right) > 0:
            kucing1 = number1_left  [-1]
            anjing1 = number1_right [-1]
            kucing2 = number2_left  [-1]
            anjing2 = number2_right [-1]

            if len(number1_left) > 3 and len(number1_right) > 3 and len(number2_left) > 3 and len(number2_right) > 3:
                number1_left .pop(0)
                number1_right.pop(0)
                number2_left .pop(0)
                number2_right.pop(0)
            
            else:
                pass

        else:
            kucing1 = 0.5
            anjing1 = 0.5
            kucing2 = 0.5
            anjing2 = 0.5

            pilihan_1 = "Invalid"   
            pilihan_2 = "Invalid"   

        if kucing1 > anjing1:
            pilihan_1 = "Kucing"

        if kucing1 < anjing1:
            pilihan_1 = "Anjing"

        if kucing1 == anjing1:
            pilihan_1 = "Invalid"

        if kucing2 > anjing2:
            pilihan_2 = "Kucing"

        if kucing2 < anjing2:
            pilihan_2 = "Anjing"

        if kucing2 == anjing2:
            pilihan_2 = "Invalid"               
        
        # Mengubah frame menjadi objek ImageTk agar dapat ditampilkan pada Label Tkinter
        img1 = Image.fromarray(rgb_frame1)
        img_tk1 = ImageTk.PhotoImage(image=img1)

        img2 = Image.fromarray(rgb_frame2)
        img_tk2 = ImageTk.PhotoImage(image=img2)

        # Menampilkan frame pertama pada elemen Label di sebelah kiri
        label_frame1.configure(image=img_tk1)

        # Menampilkan frame kedua pada elemen Label di sebelah kanan
        label_frame2.configure(image=img_tk2)

        # Memperbarui teks pada elemen Label angka pertama
        label_text1_left .configure(text="Kucing: {}".format(kucing1))
        label_text1_right.configure(text="Anjing: {}".format(anjing1))
        label_text1_center.configure(text="{}".format(pilihan_1))

        # Memperbarui teks pada elemen Label angka kedua
        label_text2_left .configure(text="Kucing: {}".format(kucing2))
        label_text2_right.configure(text="Anjing: {}".format(anjing2))
        label_text2_center.configure(text="{}".format(pilihan_2))

        # Menyimpan referensi agar gambar tidak dihapus oleh garbage collector
        label_frame1.image = img_tk1
        label_frame2.image = img_tk2

    # Memanggil fungsi ini secara berulang setiap beberapa milidetik
    label_frame1.after(5, main)

window = tk.Tk()
window.title("IMAGE CLASSIFIER DOG AND CAT")

# Membuat elemen Label untuk menampilkan frame pertama di sebelah kiri
label_frame1 = tk.Label(window)
label_frame1.grid(row=2, column=0, padx=10, pady=10)

# Membuat elemen Label untuk menampilkan angka pada frame pertama di pojok kiri atas
label_text1_left   = tk.Label(window, font=("Arial", 16), fg="green")
label_text1_left.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

label_text1_right   = tk.Label(window, font=("Arial", 16), fg="green")
label_text1_right.grid(row=1, column=0, padx=10, pady=10, sticky="ne")

label_text1_center   = tk.Label(window, font=("Arial", 16), fg="green")
label_text1_center.grid(row=0, column=0, padx=10, pady=10)

# Membuat elemen Label untuk menampilkan frame kedua di sebelah kanan
label_frame2 = tk.Label(window)
label_frame2.grid(row=2, column=1, padx=10, pady=10)

# Membuat elemen Label untuk menampilkan angka pada frame kedua di pojok kanan atas
label_text2_left   = tk.Label(window, font=("Arial", 16), fg="green")
label_text2_left.grid(row=1, column=1, padx=10, pady=10, sticky="nw")

label_text2_right   = tk.Label(window, font=("Arial", 16), fg="green")
label_text2_right.grid(row=1, column=1, padx=10, pady=10, sticky="ne")

label_text2_center   = tk.Label(window, font=("Arial", 16), fg="green")
label_text2_center.grid(row=0, column=1, padx=10, pady=10)

Thread_1 = Thread(target = pengolahan_data_kamera_depan)
Thread_1.daemon = True
Thread_2 = Thread(target = kirim_data)
Thread_2.daemon = True
Thread_3 = Thread(target = main)
Thread_3.daemon = False

while MULAI == 1:
    user = input("JALANKAN KODINGAN ATAU MATIKAN KODINGAN? (1 ATAU 0): ")
    if user == "1":
        Thread_1.start()
        Thread_2.start()
        Thread_3.start()

        window.mainloop()

    if user == "0":
        print("KODINGAN BERHASIL BERHENTI")
        sys.exit()

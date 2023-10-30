import h5py
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

def create_folder(path):
    path_folder = path
    folder_exist = os.path.exists(path_folder)
    if not folder_exist:
        os.makedirs(path_folder)
        print('create new folder'+path_folder)
    else:
        print('folder existed')

# transfer mat to jpeg
data_path = 'E:\\UAV_SIG_DATA\\'
list_file = os.listdir(data_path)
save_pic_path = 'D:\\DroneRfa\\'

Fs = 100e6  # IQ rate 100MHz
N_slice = int(10e6)  # Slice every 10M points

for file_name in list_file:
    data_file = data_path + file_name
    data = h5py.File(data_file, 'r')
    RF0_I = data['RF0_I'][0]
    RF0_Q = data['RF0_Q'][0]
    data_ch0 = RF0_I + 1j*RF0_Q #ch0: center freq 2440MHz
    RF1_I = data['RF1_I'][0]
    RF1_Q = data['RF1_Q'][0]
    data_ch1 = RF1_I + 1j*RF1_Q #ch1: center freq 5800MHz

    print("ch0 center freq: 2440MHz")
    print("ch1 center freq: 5800MHz")
    print("T10010(FrySKy X20) and T10100(Taranis Plus) ch0 center freq 915MHz, ch1 center freq 2440MHz")
    print('ch0 shape: ' +str(data_ch0.shape))
    print('ch1 shape: ' +str(data_ch1.shape))

    # create save pic folder
    pic_folder = os.path.join(save_pic_path, file_name)
    create_folder(pic_folder)


    # save pic to folder ch0
    print('slice every 10M points')
    
    pic_ch0_path = os.path.join(pic_folder, 'ch0')
    create_folder(pic_ch0_path)

    print('save ch0')
    N_loop = int(data_ch0.shape[0]/N_slice)
    for i in range(0, N_loop):
        tmp_slice_ch0 = data_ch0[i:N_slice+i]
        plt.figure(figsize=(20,8))
        plt.specgram(tmp_slice_ch0, NFFT=2048, Fs=100e6, scale='dB')
        plt.axis('off') 
        plt.xticks([])
        # pic_path = pic_ch0_path+'\\'+str(i)+'.jpeg'
        pic_path = os.path.join(pic_ch0_path, str(i)+'.jpeg')
        plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)

    pic_ch1_path = os.path.join(pic_folder, 'ch1')
    create_folder(pic_ch1_path)

    print('save ch1')
    N_loop = int(data_ch1.shape[0]/N_slice)
    for i in range(0, N_loop):
        tmp_slice_ch1 = data_ch1[i:N_slice+i]
        plt.figure(figsize=(20,8))
        plt.specgram(tmp_slice_ch1, NFFT=2048, Fs=100e6, scale='dB')
        plt.axis('off') 
        plt.xticks([])
        pic_path = os.path.join(pic_ch1_path, str(i)+'.jpeg')
        plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)

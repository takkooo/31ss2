import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from math import pi

def frequency_offset_correlation(data, Fs, offset):
    x = np.linspace(0, len(data) / Fs, len(data))
    return data * np.exp(1j * 2 * pi * offset * x)

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


Original_Fs = 100e6  # IQ rate 100MHz
Resample_rate = 50e6
factor = Original_Fs // Resample_rate
slice_time = 20e-3  # Slice every 10M points
slice_len = int(slice_time * Resample_rate)

cnt = 0


for file_name in list_file:
    type = file_name.split('_')[0]
    save_path = os.path.join(save_pic_path, type)
    create_folder(save_path)
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
    # pic_folder = os.path.join(save_pic_path, file_name)
    # create_folder(pic_folder)


    # save pic to folder ch0
    print('slice every 1M points')
    
    # pic_ch0_path = os.path.join(pic_folder, 'ch0')
    # create_folder(pic_ch0_path)

    print('save ch0')
    resample_data_ch0 = signal.resample(data_ch0, int(len(data_ch0) / factor))
    N_loop = int(len(resample_data_ch0)/slice_len)
    for i in range(0, N_loop):
        tmp_slice_ch0 = resample_data_ch0[i * slice_len:(i + 1) * slice_len]
        for ffo in range(int(-25e6), int(25e6), int(5e6)):
            cnt += 1
            new_slice_data = frequency_offset_correlation(tmp_slice_ch0, Resample_rate, ffo)
            with open(os.path.join(save_path, str(cnt)), 'w') as f:
                new_slice_data.tofile(f)
            plt.figure(figsize=(20,8))
            S_data = plt.specgram(tmp_slice_ch0, NFFT=2048, Fs=50e6, scale='dB')
            np.save(os.path.join(save_path ,'spec'+str(cnt)+'.npy'), S_data)
            plt.axis('off')
            plt.xticks([])
            # pic_path = pic_ch0_path+'\\'+str(i)+'.jpeg'
            pic_path = os.path.join(save_path, str(cnt)+'.jpeg')
            plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    print('save ch1')
    resample_data_ch1 = signal.resample(data_ch1, int(len(data_ch1) / factor))
    N_loop = int(len(resample_data_ch1) / slice_len)
    for i in range(0, N_loop):
        tmp_slice_ch1 = resample_data_ch1[i * slice_len:(i + 1) * slice_len]
        for ffo in range(int(-25e6), int(25e6), int(5e6)):
            cnt += 1
            new_slice_data = frequency_offset_correlation(tmp_slice_ch1, Resample_rate, ffo)
            with open(os.path.join(save_path, str(cnt)), 'w') as f:
                new_slice_data.tofile(f)
            plt.figure(figsize=(20, 8))
            S_data = plt.specgram(tmp_slice_ch1, NFFT=2048, Fs=50e6, scale='dB')
            np.save(os.path.join(save_path ,'spec'+str(cnt)+'.npy'), S_data)
            plt.axis('off')
            plt.xticks([])
            # pic_path = pic_ch0_path+'\\'+str(i)+'.jpeg'
            pic_path = os.path.join(save_path, str(cnt) + '.jpeg')
            S_data = plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)
            plt.close()

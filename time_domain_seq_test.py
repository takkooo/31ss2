# coding: utf-8
import os
import sys
# sys.path.append('./src')
import numpy as np
import pickle
import json
import time
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from time_utils import timestamp_to_pretty_str
from math import pi
from scipy import signal as sig
from signal_capture_for_debug import SignalCapture
from signal_capture_gpu import SignalCapture as SCGPU
from signal_capture_cupy import SignalCapture as SGCUPY

def zc_sequence(u, L):
    return np.exp(-1j * pi * u * np.arange(L) * (np.arange(1, L + 1)) / L)

def generate_time_samples(seq, Ncarriers, delta_f, Fs):
    assert len(seq) == Ncarriers
    N = int(1 / delta_f * Fs)
    x = np.zeros(N, dtype=np.complex64)
    half_carriers = Ncarriers // 2
    x[-half_carriers:] = seq[:half_carriers]
    x[:half_carriers + 1] = seq[half_carriers:]
    return np.fft.ifft(x)

def generate_time_samples_with_comb(seq, Ncarriers, delta_f, Fs, comb):
    N = int(1 / delta_f * Fs)
    x = np.zeros(N, dtype=np.complex64)
    half_carriers = Ncarriers // 2
    x[-half_carriers * comb::comb] = seq[:half_carriers]
    x[:half_carriers * comb + 1:comb] = seq[half_carriers:]
    return np.fft.ifft(x)

def frequency_offset_correct(samples, Fs, offset):
    x = np.linspace(0, len(samples) / Fs, len(samples))
    return samples * np.exp(1j * 2 * pi * offset * x)

def brute_force_search(data_list, zc_table, Fs, debug = False):
    ffo_lists = np.linspace(-15e6, 15e6, 4111) # 频域遍历， 原则上子载波间隔为15kHz，频域遍历步长不应该超过7.5kHz
    P, M, N = len(data_list), len(zc_table), len(ffo_lists) # P=4 需要遍历4个频点，M=1200 需要遍历1200条ZC序列，N=4111 需要遍历4111个频点
    band_idx, zc_root, offset, local_peak = None, None, None, None # 所在频点， 所用的序列， 所在的中心频点偏差， 当前的相关峰值
    for p in range(P):
        data = data_list[p].copy()
        acc = np.zeros((M, N))
        for m in range(M): # 码域遍历
            for n in range(N): # 频域遍历
                zc_samples = zc_table[m]
                ffo = ffo_lists[n]
                samples_with_ffo = frequency_offset_correct(zc_samples, Fs, ffo)
                tmp = np.abs(sig.correlate(data, samples_with_ffo, 'valid')) # 时域遍历
                acc[m][n] = np.max(tmp)
        if np.max(acc) > acc.mean() * 5:
            if local_peak is None or np.max(acc) > local_peak:
                u_idxs, ffo_idxs = np.where(acc == np.max(acc))
                band_idx, zc_root, offset, local_peak = p, u_idxs[0], ffo_lists[ffo_idxs[0]], np.max(acc)
    if band_idx is not None:
        '''second step'''
        data = data_list[band_idx]
        u = zc_root
        ffo = offset
        zc_samples = zc_table[u]
        samples_with_ffo = generate_time_samples(zc_samples, Fs, ffo)
        tmp = np.abs(sig.correlate(data, samples_with_ffo, 'valid'))
        if debug:
            plt.plot(tmp)
            plt.xlabel('index')
            plt.ylabel('cross_correlation peak')
            plt.show()

        peak_idx_lists, peak_height_lists = sig.find_peaks(tmp, height=np.max(tmp) / 2, distance=len(samples_with_ffo))
        if len(peak_idx_lists) > 0:
            first_peak_idx = peak_idx_lists[0]
            first_peak_height = tmp[first_peak_idx]
            return band_idx, u, ffo, first_peak_idx, first_peak_height
    return None, None, None, None, None

def prior_knowledge_search(data, zc_samples_with_ffo, pre_height = None, debug = False):
    tmp = np.abs(sig.correlate(data, zc_samples_with_ffo, 'valid'))
    if debug:
        plt.plot(tmp)
        plt.xlabel('index')
        plt.ylabel('cross_correlation peak')
        plt.show()
    if np.max(tmp) > tmp.mean() * 5:
        peak_idx_lists, peak_height_lists = sig.find_peaks(tmp, height=np.max(tmp) / 2, distance=len(zc_samples_with_ffo))
        if len(peak_idx_lists) > 0:
            first_peak_idx = peak_idx_lists[0]
            first_peak_height = tmp[first_peak_idx]
            if pre_height is None or np.max(tmp) < pre_height / 2:
                return first_peak_idx, first_peak_height

    return None, None, None, None, None


def pilot_search(file_name, Fs, chunk_len, zc_table, band_idx = None, u = None, ffo = None, debug = False):
    with open(file_name, 'rb') as f:
        data_info = pickle.load(f)
    if band_idx is None:
        data_list = list()
        for i in range(4):
            data_list.append(data_info['data_list'][i]['recv_data'][:chunk_len])
        band_idx, u, ffo, first_peak_idx, first_peak_height = brute_force_search(data_list, zc_table, Fs, debug)
    else:
        data = data_info['data_list'][band_idx]['recv_data'][:chunk_len]
        zc_samples = zc_table[u]
        samples_with_ffo = frequency_offset_correct(zc_samples, Fs, ffo)
        first_peak_idx, first_peak_height = prior_knowledge_search(data, samples_with_ffo, None, debug)
    return band_idx, u, ffo, first_peak_idx, first_peak_height



# FILE_PATH = 'D:/tdoa_exp_data/0920_exp_data/171966_data/0920_01/'
# Fs = 50e6
# signal_capture = SignalCapture(Fs)
# SCG = SCGPU(Fs)

# start_time_step = 1695193616
# # end_time_step = 1695193626
# end_time_step = 1695193617

# for time_stamp in range(start_time_step, end_time_step + 1):
#     file_name = '171966_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
#     with open(FILE_PATH + file_name, 'rb') as f:
#         data_info = pickle.load(f)
#     tmp_data = data_info['data_list'][0]['recv_data']
#     # if time_stamp == start_time_step:
#     #     with open('./test/aaa', 'wb') as f:
#     #         f.write(tmp_data)
#     #     break

#     raw_data = [tmp_data]
#     # time1 = time.time()
#     # flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa = signal_capture.blind_search(raw_data)
#     # if flag:
#     #     print(flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa)
#     # time2 = time.time()
#     # print(time_stamp, time2 - time1)

#     time1 = time.time()
#     flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa = SCG.blind_search(raw_data)
#     if flag:
#         print(flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa)
#     time2 = time.time()
#     print(time_stamp, time2 - time1)




file_name = 'data\\180007_1695193424_2023-09-20_15-03-44.pkl'
Fs = 50e6
signal_capture = SignalCapture(Fs)
SCG = SGCUPY(Fs)
with open(file_name, 'rb') as f:
    data_info = pickle.load(f)

tmp_data = data_info['data_list'][0]['recv_data']

raw_data = tmp_data.copy()

time1 = time.time()
flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa = SCG.blind_search(raw_data)
if flag:
    print(flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa)
time2 = time.time()
print(time2 - time1)

file_name = 'data\\180007_1695193424_2023-09-20_15-03-44.pkl'
Fs = 50e6
signal_capture = SignalCapture(Fs)
SCG = SGCUPY(Fs)
with open(file_name, 'rb') as f:
    data_info = pickle.load(f)

tmp_data = data_info['data_list'][0]['recv_data']

raw_data = [tmp_data]

time1 = time.time()
flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa = signal_capture.blind_search(raw_data)
if flag:
    print(flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa)
time2 = time.time()
print(time2 - time1)













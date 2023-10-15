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
from signal_capture_cupy import SignalCapture as SCGPU

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



FILE_PATH = 'D:/tdoa_exp_data/0920_exp_data/171966_data/0920_01/'
Fs = 50e6
signal_capture = SignalCapture(Fs)
SCG = SCGPU(Fs)

start_time_step = 1695193616
end_time_step = 1695193626
# end_time_step = 1695193617

for time_stamp in range(start_time_step, end_time_step + 1):
    file_name = '171966_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
    with open(FILE_PATH + file_name, 'rb') as f:
        data_info = pickle.load(f)
    tmp_data = data_info['data_list'][0]['recv_data']
    # if time_stamp == start_time_step:
    #     with open('./test/aaa', 'wb') as f:
    #         f.write(tmp_data)
    #     break

    raw_data = tmp_data.copy()
    # time1 = time.time()
    # flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa = signal_capture.blind_search(raw_data)
    # if flag:
    #     print(flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa)
    # time2 = time.time()
    # print(time_stamp, time2 - time1)

    time1 = time.time()
    flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa = SCG.blind_search(raw_data)
    if flag:
        print(flag, packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa)
    time2 = time.time()
    print(time_stamp, time2 - time1)







# Fs = 50e6
# chunk_length = int(Fs * 10e-3)
# delta_f = 15e3
# Ncarriers = 1201
# SERIALS = ['171939', '171966', '180000', '180007']
# signal_capture = SignalCapture(Fs, True)

# start_time_stamp = 1691569557
# end_time_stamp = 1691570110
#
# L = 1201
# u = 601
# zc_seq = zc_sequence(u, L)
# zc_samples = generate_time_samples(zc_seq, L, delta_f, Fs)
# zc_samples = np.append(zc_samples[-int(72/1024/delta_f * Fs):], zc_samples)
# for time_stamp in range(start_time_stamp, start_time_stamp + 120, 10):
#     try:
#         for serial in SERIALS:
#             file_name = serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
#             with open(os.path.join(FILE_PATH, file_name), 'rb') as f:
#                 data_info = pickle.load(f)
#                 data = data_info['data_list'][0]['recv_data'].copy()
#                 # with open('./recv_space/0809_exp/binary_file/' + file_name[:17], 'wb') as f:
#                 #     f.write(data)
#             tmp = np.abs(sig.correlate(data, zc_samples, 'valid'))
#             plt.plot(tmp)
#             plt.title(serial + '_' + str(time_stamp))
#             plt.show()
#     except Exception as e:
#         print(e)
#         continue



# L = 1201
# u = 601
# zc_seq = zc_sequence(u, L)
# zc_samples = generate_time_samples(zc_seq, L, delta_f, Fs)
# # zc_samples = np.append(zc_samples[-int(72/1024/delta_f * Fs):], zc_samples)
# with open('./recv_space/0815_exp/binary_file/180007_1692091016', 'rb') as f:
#     data = np.fromfile(f, dtype="<f").astype(np.float32).view(np.complex64)
#
# for ffo in [-20e6, -10e6, 0, 10e6, 20e6]:
#     zc_samples_ffo = frequency_offset_correct(zc_samples, Fs, ffo)
#     tmp = np.abs(sig.correlate(data, zc_samples_ffo, 'valid'))
#     plt.plot(tmp)
#     plt.show()


# L = 601
# zc_max = np.zeros(L)
# for u in range(1, L):
#     zc_seq = zc_sequence(u, L)
#     zc_samples = generate_time_samples(zc_seq, L, delta_f * 2, Fs)
#     # zc_samples = np.append(zc_samples, zc_samples)
#     tmp = np.abs(sig.correlate(data, zc_samples, 'valid'))
#     zc_max[u - 1] = np.max(tmp)
# plt.plot(zc_max)
# plt.show()
# print(np.where(zc_max > 0.8 * np.max(zc_max)), np.max(zc_max))

# u1, u2 = 601, 1199
# L = 1201
# zc1 = zc_sequence(u1, L)
# zc2 = zc_sequence(u2, L)
# zc_samples1 = generate_time_samples(zc1, L, delta_f * 1, Fs)
# zc_samples2 = generate_time_samples(zc2, L, delta_f * 1, Fs)
# zc_samples = zc_samples1
# tmp = np.abs(sig.correlate(data, zc_samples, 'valid'))
# plt.plot(tmp)
# plt.show()




######### uav signal filter ###############
# band_idx_list, u_list, ffo_list = [], [], []
# sig_filter_res = dict()
# for time_stamp in range(start_time_stamp, end_time_stamp + 1):
#     try:
#         for serial in ['180007', '171966', '180000', '171939']:
#             file_name = serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
#             with open(os.path.join(FILE_PATH, file_name), 'rb') as f:
#                 data_info = pickle.load(f)
#             raw_data = list()
#             for i in range(4):
#                 raw_data.append(data_info['data_list'][i]['recv_data'][:chunk_length])
#             if len(band_idx_list) == 0:
#                 flag, band_idx, u, ffo = signal_capture.find_uav_sig(raw_data, ['video_20M'])
#             else:
#                 band_idx, u, ffo = band_idx_list[-1], u_list[-1], ffo_list[-1]
#                 flag, band_idx, u, ffo = signal_capture.find_uav_sig(raw_data, ['video_20M'], band_idx, u, ffo)
#             if flag:
#                 band_idx_list.append(band_idx)
#                 u_list.append(u)
#                 ffo_list.append(ffo)
#                 break
#         sig_filter_res[time_stamp] = [flag, int(band_idx), int(u), ffo]
#         print(serial, time_stamp, flag, band_idx, u, ffo)
#
#     except Exception as e:
#         print(e)
#         continue

# with open('./log/0609_exp_sig_filter_res.json', 'w') as f:
#     json.dump(sig_filter_res, f)


################ ToA calculate #################
# with open('./log/0609_exp_sig_filter_res.json', 'r') as f:
#     sig_filter_res = json.load(f)
#
# ToA_res = dict()
# for time_stamp in range(start_time_stamp, end_time_stamp + 1):
#     try:
#         tmp_toa = []
#         if str(time_stamp) in sig_filter_res:
#             flag, band_idx, u, ffo = sig_filter_res[str(time_stamp)]
#             u = 1
#             if flag:
#
#                 for serial in ['180007']:
#                     file_name = serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
#                     with open(os.path.join(FILE_PATH, file_name), 'rb') as f:
#                         data_info = pickle.load(f)
#                     data = data_info['data_list'][band_idx]['recv_data'][:chunk_length]
#                     zc_seq = zc_sequence(u, Ncarriers)
#                     zc_samples = generate_time_samples(zc_seq, Ncarriers, delta_f, Fs)
#                     samples_with_ffo = frequency_offset_correct(zc_samples, Fs, ffo)
#                     tmp = np.abs(sig.correlate(data, samples_with_ffo, 'valid'))
#                     if np.max(tmp) > tmp.mean() * 10:
#                         # peak_idx_lists, peak_height_lists = sig.find_peaks(tmp, height=np.max(tmp) * 0.8, distance=len(samples_with_ffo))
#                         # if len(peak_idx_lists) > 0:
#                         #     tmp_toa.append({serial: list(peak_idx_lists)})
#                         plt.plot(tmp)
#
#
#                         plt.xlabel(str(time_stamp))
#                         plt.show()
#                         with open('./recv_space/0609_exp/binary_file/' + file_name[:17], 'wb') as f:
#                             f.write(data)
#         print(tmp_toa)
#         ToA_res[time_stamp] = tmp_toa
#     except Exception as e:
#         print(e)
#         continue

# with open('./log/0609_exp_toa_res.json', 'w') as f:
#     json.dump(ToA_res, f)










# zc_table = np.zeros((Ncarriers - 1, int(1 / delta_f * Fs)), dtype=np.complex64)
# for u in range(1, Ncarriers):
#     zc_seq = zc_sequence(u, Ncarriers)
#     samples = generate_time_samples(zc_seq, Ncarriers, delta_f, Fs)
#     # resample_samples = np.interp(np.arange(0, len(samples), origi_fs / Fs), np.arange(0, len(samples)),
#     #                                 samples)
#     zc_table[u - 1][:] = samples.copy()
#
#
# start_time_stamp = 1690186380
# end_time_stamp = 1690186380
# SERIALS = ['171939', '171966', '180000', '180007']
# # band_idx = 2
# # ffo = 11659491.193737768
# # u = 600
# #
# # res_filter = dict()
# # for time_stamp in range(start_time_stamp, end_time_stamp + 2, 2):
# #     for serial in SERIALS:
# #         # file_name = FILE_PATH + serial + '_data/' + '0724_02/' + serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
# #         file_name = './test/' + serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
# #         band_idx, u, ffo, first_peak_idx, first_peak_height = pilot_search(file_name, Fs, chunk_length, band_idx, u, ffo, True)
# #         print(first_peak_idx)
#
#
#
#
#
#
# # u = 807
# # ffo = -14547945.205479452
# # band_idx = 0
# signal_capture = SignalCapture(Fs, debug = True)
#
#
# for time_stamp in range(start_time_stamp, end_time_stamp + 2, 2):
#     for serial in SERIALS:
#         # file_name = FILE_PATH + serial + '_data/' + '0724_02/' + serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
#         file_name = './test/' + serial + '_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl'
#         # band_idx, u, ffo, first_peak_idx, first_peak_height = pilot_search(file_name, Fs, chunk_length, band_idx, u, ffo, debug=True)
#         # print(band_idx, u, ffo, first_peak_idx, first_peak_height)
#
#         with open(file_name, 'rb') as f:
#             data_info = pickle.load(f)
#         data = data_info['data_list'][1]['recv_data'][:chunk_length]
#         ffo = 4504394.53125
#         idx_list = []
#         for u in [0, Ncarriers - 2]:
#             zc_samples = zc_table[u]
#             samples_with_ffo = frequency_offset_correct(zc_samples, Fs, ffo)
#             tmp = np.abs(sig.correlate(data, samples_with_ffo, 'valid'))
#             # plt.plot(tmp)
#             # plt.title(serial + str(u))
#             # plt.show()
#             idx = np.argmax(tmp)
#             idx_list.append(idx)
#         print(serial, idx_list)
#
#
#
#
#
#
#
#         # with open(file_name, 'rb') as f:
#         #     data_info = pickle.load(f)
#         # for i in range(4):
#         #     data = data_info['data_list'][i]['recv_data']
#         #     with open('./recv_space/0609_exp/binary_file/' + file_name[7:24] + ' ' + str(i + 1), 'wb') as f:
#         #         f.write(data)
#         # sys.exit()
#
#         # zc_samples = zc_table[807]
#         # ffo_lists = np.linspace(-15e6, 15e6, 5111)
#         # with open(file_name, 'rb') as f:
#         #     data_info = pickle.load(f)
#         # start_time = time.time()
#         # band_idx, offset, loc_peak = None, None, None
#         # for i in range(4):
#         #     data = data_info['data_list'][i]['recv_data'][:chunk_length]
#         #     ffo_acc = np.zeros(len(ffo_lists))
#         #     for j in range(len(ffo_lists)):
#         #         ffo = ffo_lists[j]
#         #         samples_with_ffo = frequency_offset_correct(zc_samples, Fs, ffo)
#         #         tmp = np.abs(sig.correlate(data, samples_with_ffo, 'valid'))
#         #         ffo_acc[j] = np.max(tmp)
#         #     if loc_peak is None or np.max(ffo_acc) > loc_peak:
#         #         band_idx = i
#         #         ffo_idx = np.argmax(ffo_acc)
#         #         offset = ffo_lists[ffo_idx]
#         #         loc_peak = np.max(ffo_acc)
#         #     plt.subplot(2, 2, i + 1)
#         #     plt.plot(ffo_acc)
#         # plt.show()
#         # end_time = time.time()
#         # print(band_idx, offset, end_time - start_time)
#         # data = data_info['data_list'][band_idx]['recv_data'][:chunk_length]
#         # samples_with_ffo = frequency_offset_correct(zc_samples, Fs, offset)
#         # tmp = np.abs(sig.correlate(data, samples_with_ffo, 'valid'))
#         # plt.plot(tmp)
#         # plt.show()


# time_stamp = 1686304722
# with open('/Volumes/新加卷/0609_real_01_1686303834_save_data/' + '180007_' + str(time_stamp) + '_' + timestamp_to_pretty_str(time_stamp) + '.pkl', 'rb') as f:
#     data_info = pickle.load(f)
#     for i in range(4):
#         data = data_info['data_list'][i]['recv_data']
#         with open('./recv_space/0609_exp/binary_file/' + str(i), 'wb') as f:
#             f.write(data)















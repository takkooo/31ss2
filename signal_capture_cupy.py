# coding: utf-8
import numpy as np
import cupy as np_gpu
import sys
# import matplotlib
# import scipy
from scipy import signal
# from matplotlib import pyplot as plt
from math import pi
import timeit
from cupyx import scipy as scipy_gpu
from cupyx.scipy import signal as signal_gpu

'''
用于检测大疆无人机图传信号并且计算到达时间（time of arrival）
主程序 find_uav_sig 
输入：raw_data：[[第0信道采样数据], [第1信道采样数据], ..., [第N信道采样数据]]； 
     candidate_type: 从['video_10Mz'，'video_20MHz'，'video_40Mz']选择要监控的信道大小，例如要监控20MHz图传，选择['video_20MHz']
     freq_band_idx: 指示图传信号所在信道的已知信息；
     zc_root：指示图传信号所用zc序列的已知信息; 
     freq_offset：指示图传信号相对中心频点偏移的已知信息
输出：flag: 用于指示是否识别到无人机信号；
     freq_band_idx: 用于指示无人机图传信号在哪个信道；
     u: 无人机图传所使用的ZC序列的根；
     ffo：无人机信号所在信道相较于中心频点的频偏；
     toa: 到达时间
'''

class SignalCapture():
    def __init__(self, sample_date, debug = False):
        self.candidate_types = ['video_10M', 'video_20M', 'video_40M']
        self.NFFT = {'video_10M': 1024, 'video_20M': 2048, 'video_40M': 4096}
        self.NCARRIERS = {'video_10M': 601, 'video_20M': 1201, 'video_40M': 2401}
        self.CHUNK_LENGTH = 20e-3
        self.DELTA_F = 15e3
        self.Fs = int(sample_date)
        self.debug = debug

    def find_uav_sig(self, raw_data, packet_type = None, freq_band_idx = None, zc_root = None, freq_offset = None):
        if packet_type is not None and freq_band_idx is not None and zc_root is not None and freq_offset is not None:
            flag, band_idx, u, ffo, toa = self.priori_search(raw_data, packet_type, freq_band_idx, zc_root, freq_offset)
            if flag:
                return flag, packet_type, band_idx, u, ffo, toa
        flag, packet_type, band_idx, u, ffo, toa = self.blind_search(raw_data)
        if flag:
            return flag, packet_type, band_idx, u, ffo, toa
        return False, None, None, None, None, None

    def priori_search(self, raw_data, packet_type, freq_band_idx, zc_root, freq_offset):
        # recv_data = raw_data.copy()
        recv_data = np_gpu.asarray(raw_data)
        chunk_len = int(self.Fs * self.CHUNK_LENGTH)
        if len(recv_data) < chunk_len:
            return False, None, None, None, None
        Ncarriers = self.NCARRIERS[packet_type]
        u = zc_root
        ffo = freq_offset
        zc_seq = self.zc_sequence(u, Ncarriers)
        zc_samples = self.generate_time_samples(zc_seq, Ncarriers, self.DELTA_F, self.Fs)
        zc_seq_with_ffo = self.frequency_offset_correct(zc_samples, self.Fs, ffo)
        tmp = np_gpu.abs(signal_gpu.correlate(recv_data[:chunk_len], zc_seq_with_ffo, 'valid'))
        tmp_cpu = tmp.get()
        if np_gpu.max(tmp) > tmp.mean() * 2:
            peaks, _ = signal.find_peaks(tmp, height=np.max(tmp) / np.sqrt(2), distance=len(zc_samples))
            if len(peaks) == 1:
                return True, freq_band_idx, u, ffo, peaks[0]
        return False, None, None, None, None

    def blind_search(self, raw_data):
        candi_packet_type, freq_band_idx, cur_u, coarse_ffo, cur_peak, cur_toa = None, None, None, None, 0, None
        chunk_len = int(self.Fs * self.CHUNK_LENGTH)
        for packet_type in self.candidate_types:
            Ncarriers = self.NCARRIERS[packet_type]
#            for i in range(len(raw_data)):
            # recv_data = raw_data.copy()
            recv_data = np_gpu.asarray(raw_data)
            if len(recv_data) < chunk_len:
                continue
            for u in [1, Ncarriers - 1, (1 + Ncarriers) // 2]:
                zc_seq = self.zc_sequence(u, Ncarriers)
                zc_samples = self.generate_time_samples(zc_seq, Ncarriers, self.DELTA_F, self.Fs)
                for ffo in [-20e6, -10e6, 0, 10e6, 20e6]:
                    zc_samples_with_ffo = self.frequency_offset_correct(zc_samples, self.Fs, ffo)
                    tmp = np_gpu.abs(signal_gpu.correlate(recv_data[:chunk_len], zc_samples_with_ffo, 'valid'))
                    if np_gpu.max(tmp) > tmp.mean() * 2:
                        tmp_cpu = tmp.get()
                        peaks, _ = signal.find_peaks(tmp_cpu, height=np.max(tmp_cpu) / np.sqrt(2), distance=len(zc_samples))
                        if len(peaks) == 1 and tmp_cpu[peaks[0]] > cur_peak:
                            candi_packet_type, freq_band_idx, cur_u, coarse_ffo, cur_peak, cur_toa = packet_type, 1, u, ffo, tmp[peaks[0]], peaks[0]
                                # if self.debug:
                                #     plt.plot(tmp)
                                #     plt.title(candi_packet_type + '_' + str(cur_u) + '_' + str(coarse_ffo))
                                #     plt.show()
        if freq_band_idx is not None:
            recv_data = np_gpu.asarray(raw_data)
            Ncarriers = self.NCARRIERS[candi_packet_type]
            zc_seq = self.zc_sequence(cur_u, Ncarriers)
            zc_samples = self.generate_time_samples(zc_seq, Ncarriers, self.DELTA_F, self.Fs)
            for ffo in range(int(coarse_ffo - 10e6), int(coarse_ffo + 10e6), int(2e6)):
                zc_samples_with_ffo = self.frequency_offset_correct(zc_samples, self.Fs, ffo)
                tmp = np_gpu.abs(signal_gpu.correlate(recv_data[:chunk_len], zc_samples_with_ffo, 'valid'))
                if np_gpu.max(tmp) > cur_peak:
                    coarse_ffo, cur_peak = ffo, np_gpu.max(tmp)
            return True, candi_packet_type, freq_band_idx, cur_u, coarse_ffo, cur_toa
        else:
            return False, None, None, None, None, None

    def zc_sequence(self, u, L):
        return np_gpu.exp(-1j * pi * u * np_gpu.arange(L) * (np_gpu.arange(1, L + 1)) / L)

    def generate_time_samples(self, seq, Ncarriers, delta_f, Fs):
        assert len(seq) == Ncarriers
        N = int(1 / delta_f * Fs)
        x = np_gpu.zeros(N, dtype=np_gpu.complex64)
        half_carriers = Ncarriers // 2
        x[-half_carriers:] = seq[:half_carriers]
        x[:half_carriers + 1] = seq[half_carriers:]
        tmp = np_gpu.fft.ifft(x)
        return tmp / np_gpu.linalg.norm(tmp)

    def frequency_offset_correct(self, samples, Fs, offset):
        x = np_gpu.linspace(0, len(samples) / Fs, len(samples))
        return samples * np_gpu.exp(1j * 2 * pi * offset * x)


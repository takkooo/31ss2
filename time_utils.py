# coding:utf-8
import time


def str2timestamp(time_str):
    time_array = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_array))
    return time_stamp


def timestamp_to_str(time_stamp):
    time_array = time.localtime(time_stamp)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    return time_str

def timestamp_to_pretty_str(time_stamp):
    time_array = time.localtime(time_stamp)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time_array)
    return time_str

def pretty_str_to_timestamp(time_str):
    time_array = time.strptime(time_str, "%Y-%m-%d_%H:%M:%S")
    time_stamp = int(time.mktime(time_array))
    return time_stamp

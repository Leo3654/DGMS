#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/12/22 19:55
# @Author   : Xiang Ling
# @File     : utils.py
# @Lab      : nesa.zju.edu.cn
# ************************************
import os
import matplotlib.pyplot as plt
from datetime import datetime

from texttable import Texttable


def chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def arguments_to_tables(args):
    """
    util function to print the logs in table format
    :param args: parameters
    :return:
    """
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_precision(width=10)
    table.set_cols_dtype(['t', 't'])
    table.set_cols_align(['l', 'l'])
    table.add_rows([["Parameter", "Value"]])
    for k in keys:
        table.add_row([k, args[k]])
    return table.draw()


def int_2_one_hot(n, n_classes):
    v = [0] * n_classes
    v[n] = 1
    return v

def draw_loss(file_path, train_batch_size, train_sample_size):
    train_losses = []
    train_counter = []
    val_losses = []
    val_counter = []
    time_elapsed = []
    
    
    my_file = open(file_path, 'r')
    lines = my_file.readlines()
    for line in lines:
        if line[:6] == '#Valid':
            print(line.split(':')[1].split('#') [1])
            val_losses.append(float(line.split(':')[1].split('#')[1]))
            val_counter.append(float(line.split(':')[0].split(' ')[-1]) \
                                     * train_batch_size / train_sample_size)
        if line[:6] == '@Train':
            train_losses.append(float(line.split(':')[1].split('@')[1]))
            epoch_count = float(line.split(':')[0].split(' ')[-1]) \
                                     * train_batch_size / train_sample_size
            train_counter.append(epoch_count)
            minute = float(line.split('=')[-1].split(':')[1])
            second = float(line.split('=')[-1].split(':')[2].split('.')[0])
            time_elapsed.append(minute+second/60)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(val_counter, val_losses, color='red')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    plt.xlabel('number of epochs')
    plt.ylabel('triplet similarity loss')
    plt.savefig('curve.png')
    fig = plt.figure()
    plt.plot(train_counter, time_elapsed, color='blue')
    #plt.scatter(val_counter, val_losses, color='red')
    plt.legend(['Time Elapsed'], loc='upper right')
    plt.xlabel('number of epochs')
    plt.ylabel('time elapsed for every 2000 iterations (min)')
    plt.savefig('time.png')


#draw_loss('../PythonLogs/2023-04-03@17:43:34/log_Filter_100_1CONV_rgcn_2MATCH_submul_3MatchAgg_fc_max_margin_0.5_4MaxIter_312189_trainBS_10_validBS_50_LR_0.0001_Dropout_0.1.txt', 10, 312189)

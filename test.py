#   MIT License

# Copyright (c) 2019 Bill Hsieh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import wave as we
import numpy as np
import mir_eval
import csv
import re
from model import Net
from utils import melody_eval,est,seg,iseg


npz_dir = "./NPZ/"
train_label_dir = './GT/'



Net = Net()
Net.cuda()
Net.float()
Net.train()
GPU = True

params = list(Net.parameters())
k = 0
for i in params:
    l = 1
    print("Structure of this layer:" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("Parameters of this layer:" + str(l))
    k = k + l
print("Total Parameters: " + str(k))

Net.load_state_dict(torch.load('train_single_for_pitch_epoch_22.pkl')) 



testaudioname_list = []
dataset_title = ['MIR1K','ADC2004','MIREX2005','IK_test']
#MIR1K
mir1k_testaudioname_list = ['leon_1_01', 'leon_1_02', 'leon_1_03', 'leon_1_04', 'leon_1_05', 'leon_1_06', 'leon_1_07', 'leon_1_08', 'leon_1_09', 'leon_1_10', 'leon_1_11', 'leon_1_12', 'leon_2_01', 'leon_2_02', 'leon_2_03', 'leon_2_04', 'leon_2_05', 'leon_2_06', 'leon_2_07', 'leon_2_08', 'leon_2_09', 'leon_2_10', 'leon_2_11', 'leon_3_01', 'leon_3_02', 'leon_3_03', 'leon_3_04', 'leon_3_05', 'leon_3_06', 'leon_3_07', 'leon_3_08', 'leon_3_09', 'leon_3_10', 'leon_3_11', 'leon_3_12', 'leon_3_13', 'leon_4_01', 'leon_4_02', 'leon_4_03', 'leon_4_04', 'leon_4_05', 'leon_4_06', 'leon_4_07', 'leon_4_08', 'leon_5_01', 'leon_5_02', 'leon_5_03', 'leon_5_04', 'leon_5_05', 'leon_5_06', 'leon_5_07', 'leon_5_08', 'leon_5_09', 'leon_5_10', 'leon_5_11', 'leon_5_12', 'leon_6_01', 'leon_6_02', 'leon_6_03', 'leon_6_04', 'leon_6_05', 'leon_6_06', 'leon_6_07', 'leon_6_08', 'leon_6_09', 'leon_7_01', 'leon_7_02', 'leon_7_03', 'leon_7_04', 'leon_7_05', 'leon_7_06', 'leon_7_07', 'leon_7_08', 'leon_7_09', 'leon_7_10', 'leon_7_11', 'leon_7_12', 'leon_7_13', 'leon_8_01', 'leon_8_02', 'leon_8_03', 'leon_8_04', 'leon_8_05', 'leon_8_06', 'leon_8_07', 'leon_8_08', 'leon_8_09', 'leon_8_10', 'leon_8_11', 'leon_8_12', 'leon_8_13', 'leon_9_01', 'leon_9_02', 'leon_9_03', 'leon_9_04', 'leon_9_05', 'leon_9_06', 'stool_1_01', 'stool_1_02', 'stool_1_03', 'stool_1_04', 'stool_1_05', 'stool_1_06', 'stool_1_07', 'stool_1_08', 'stool_1_09', 'stool_2_01', 'stool_2_02', 'stool_2_03', 'stool_2_04', 'stool_2_05', 'stool_2_06', 'stool_2_07', 'stool_2_08', 'stool_3_01', 'stool_3_02', 'stool_3_03', 'stool_3_04', 'stool_3_05', 'stool_3_06', 'stool_3_07', 'stool_3_08', 'stool_3_09', 'stool_3_10', 'stool_4_01', 'stool_4_02', 'stool_4_03', 'stool_4_04', 'stool_4_05', 'stool_4_06', 'stool_4_07', 'stool_4_08', 'stool_4_09', 'stool_4_10', 'stool_5_01', 'stool_5_02', 'stool_5_03', 'stool_5_04', 'stool_5_05', 'stool_5_06', 'stool_5_07', 'stool_5_08', 'tammy_1_01', 'tammy_1_02', 'tammy_1_03', 'tammy_1_04', 'tammy_1_05', 'tammy_1_06', 'tammy_1_07', 'tammy_1_08', 'titon_1_01', 'titon_1_02', 'titon_1_03', 'titon_1_04', 'titon_1_05', 'titon_1_06', 'titon_1_07', 'titon_1_08', 'titon_2_01', 'titon_2_02', 'titon_2_03', 'titon_2_04', 'titon_2_05', 'titon_2_06', 'titon_2_07', 'titon_2_08', 'titon_2_09', 'titon_3_01', 'titon_3_02', 'titon_3_03', 'titon_3_04', 'titon_3_05', 'titon_3_06', 'titon_3_07', 'titon_3_08', 'titon_4_01', 'titon_4_02', 'titon_4_03', 'titon_4_04', 'titon_4_05', 'titon_4_06', 'titon_4_07', 'titon_4_08', 'titon_4_09', 'titon_4_10', 'titon_4_11', 'titon_5_01', 'titon_5_02', 'titon_5_03', 'titon_5_04', 'titon_5_05', 'titon_5_06', 'titon_5_07', 'titon_5_08', 'titon_5_09', 'yifen_1_01', 'yifen_1_02', 'yifen_1_03', 'yifen_1_04', 'yifen_1_05', 'yifen_1_06', 'yifen_1_07', 'yifen_1_08', 'yifen_1_09', 'yifen_1_10', 'yifen_1_11', 'yifen_1_12', 'yifen_1_13', 'yifen_1_14', 'yifen_1_15', 'yifen_1_16', 'yifen_2_01', 'yifen_2_02', 'yifen_2_03', 'yifen_2_04', 'yifen_2_05', 'yifen_2_06', 'yifen_2_07', 'yifen_2_08', 'yifen_2_09', 'yifen_2_10', 'yifen_2_11', 'yifen_2_12', 'yifen_2_13', 'yifen_2_14', 'yifen_2_15', 'yifen_3_01', 'yifen_3_02', 'yifen_3_03', 'yifen_3_04', 'yifen_3_05', 'yifen_3_06', 'yifen_3_07', 'yifen_3_08', 'yifen_3_09', 'yifen_3_10', 'yifen_3_11', 'yifen_3_12', 'yifen_4_01', 'yifen_4_02', 'yifen_4_03', 'yifen_4_04', 'yifen_4_05', 'yifen_4_06', 'yifen_4_07', 'yifen_4_08', 'yifen_4_09', 'yifen_4_10', 'yifen_4_11', 'yifen_5_01', 'yifen_5_02', 'yifen_5_03', 'yifen_5_04', 'yifen_5_05', 'yifen_5_06', 'yifen_5_07', 'yifen_5_08', 'yifen_5_09', 'yifen_5_10', 'yifen_5_11']
testaudioname_list.append(mir1k_testaudioname_list)
#ADC2004
adc2004_testaudioname_list = ['daisy1', 'daisy2', 'daisy3', 'daisy4', 'opera_fem2', 'opera_fem4', 'opera_male3', 'opera_male5', 'pop1', 'pop2', 'pop3', 'pop4']
testaudioname_list.append(adc2004_testaudioname_list)
#MIREX2005
mirex2005_testaudioname_list = ['train01', 'train02', 'train03', 'train04', 'train05', 'train06', 'train07', 'train08', 'train09']
testaudioname_list.append(mirex2005_testaudioname_list)
#IK_test
ik_testaudioname_list = ['54242_verse', '54243_chorus', '54243_verse', '54245_chorus', '54245_verse', '54246_chorus', '54246_verse', '54247_verse', '54249_chorus', '54249_verse', '54251_verse', '61647_chorus', '61647_verse', '61670_verse', '61671_chorus', '61671_verse', '61673_verse', '61674_verse', '61676_chorus', '61677_chorus', '66556_chorus', '66556_verse', '66557_chorus', '66558_verse', '66559_chorus', '66559_verse', '66560_verse', '66561_chorus', '66563_chorus', '66563_verse', '66564_verse', '66565_verse', '66566_verse', '71706_verse', '71710_chorus', '71710_verse', '71711_chorus', '71711_verse', '71712_verse', '71716_chorus', '71719_chorus', '71719_verse', '71720_chorus', '71720_verse', '71723_verse', '71726_chorus', '80612_verse', '80614_verse', '80616_verse', '90586_chorus', '90586_verse', '90587_verse']
testaudioname_list.append(ik_testaudioname_list)

for i,listname in enumerate(testaudioname_list):
    print(dataset_title[i])
    print(listname)

    csvfile2 = open('./test_result_of_' + dataset_title[i] + '.csv', 'w', newline='') 
    writer2 = csv.writer(csvfile2)
    writer2.writerow(['Songname', 'VR', 'VFA', 'RPA', 'RCA', 'OA'])
    avg_arr = [0,0,0,0,0]
    for name in listname:
        A = np.load(npz_dir+name+'.npz')
        data = A['data']
        time_arr = A['time_arr']
        CenFreq = A['CenFreq']
        CenFreq[0] = 0
        f2 = open(train_label_dir + name + "REF_label.txt",'r')
        gt_label = f2.read()
        gt_label = eval(gt_label)
        f2.close()
        gt_label = np.array(gt_label,dtype = np.int64)
        gt_label_ts = torch.from_numpy(gt_label)
        frames = data.shape[-1]
        if frames > 3120:
            sx, sy, listlen = seg(data,gt_label_ts,3120)              
            seg_y = []
            for i in range(listlen):
                batch_x = sx[i]
                
                gt_label_ts_cut = sy[i]
                batch_x = batch_x[np.newaxis,:]
                batch_x = torch.from_numpy(batch_x).float()
                if GPU:
                    batch_x = batch_x.cuda()
                pred_y = Net(batch_x)
                pred_y = pred_y.cpu().detach().numpy()
                seg_y.append(pred_y)
            
            pred_y = iseg(seg_y)
        else:
            batch_x = data[np.newaxis,:]
            batch_x = torch.from_numpy(batch_x).float()
            if GPU:
                batch_x = batch_x.cuda()
            pred_y = Net(batch_x)             
            pred_y = pred_y.cpu().detach().numpy()


        ref_arr = np.loadtxt(train_label_dir + name + "REF.txt")
        est_arr = est(pred_y, CenFreq, time_arr)
        eval_arr = melody_eval(ref_arr, est_arr)
        avg_arr += eval_arr

        writer2.writerow([name, eval_arr[0], eval_arr[1], eval_arr[2], eval_arr[3], eval_arr[4]]) 

    csvfile2.close()


avr_root_list = []
avr_majmin_list = []
avr_overseg_list = []
avr_underseg_list = []
avr_seg_list = []
for i,listname in enumerate(testaudioname_list):
    with open('./test_result_of_' + dataset_title[i] + '.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        count = 0
        root_list = []
        majmin_list = []
        overseg_list = []
        underseg_list = []
        seg_list = []

        for row in f_csv:
            root_list.append(eval(row[1]))
            majmin_list.append(eval(row[2]))
            overseg_list.append(eval(row[3]))
            underseg_list.append(eval(row[4]))
            seg_list.append(eval(row[5]))
            count += 1
    avr_root = sum(root_list)/len(root_list)
    avr_majmin =  sum(majmin_list)/len(root_list)
    avr_overseg =  sum(overseg_list)/len(root_list)
    avr_underseg =  sum(underseg_list)/len(root_list)
    avr_seg =  sum(seg_list)/len(root_list)

    avr_root_list.append(avr_root)
    avr_majmin_list.append(avr_majmin)
    avr_overseg_list.append(avr_overseg)
    avr_underseg_list.append(avr_underseg)
    avr_seg_list.append(avr_seg)

with open('./test_result_total.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Dataset', 'VR', 'VFA', 'RPA', 'RCA', 'OA'])
                    for i,listname in enumerate(testaudioname_list):
                        writer.writerow([dataset_title[i], avr_root_list[i], avr_majmin_list[i], avr_overseg_list[i], avr_underseg_list[i], avr_seg_list[i]])
                    
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

def melody_eval(ref, est):
    ref_time = ref[:,0]
    ref_freq = ref[:,1]
    est_time = est[:,0]
    est_freq = est[:,1]
    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr


def est(output, CenFreq, time_arr):
    
    CenFreq[0] = 0
    est_time = time_arr
    output = output[0,0,:,:]
    est_freq = np.argmax(output, axis=0)
    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]      
    est_arr = np.concatenate((est_time[:,None],est_freq[:,None]),axis=1)
    return est_arr


def seg(data,label,seg_frames_length=3120):
    frames = data.shape[-1]
    cutnum = int(frames / seg_frames_length)
    remain = frames - (cutnum*seg_frames_length)
    xlist = []
    ylist = []
    for i in range(cutnum):
        x = data[:,:, i*seg_frames_length:(i+1)*seg_frames_length]
        y = label[i*seg_frames_length:(i+1)*seg_frames_length]
        xlist.append(x)
        ylist.append(y)
    if frames % seg_frames_length != 0:
        x = data[:,:, cutnum*seg_frames_length:]
        y = label[cutnum*seg_frames_length:]
        xlist.append(x)
        ylist.append(y)
    return xlist,ylist,len(xlist)
def iseg(data, seg_frames_length=256):
    x = data[0]
    for i in range(len(data)-1):
        x = np.concatenate((x, data[i+1]), axis=-1)
    return x
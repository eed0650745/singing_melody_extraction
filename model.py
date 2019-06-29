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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_d_3 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 10, 3, padding=3,dilation=3),
            nn.SELU()
            )
        self.conv_d_6 = nn.Sequential(
            nn.BatchNorm2d(13),
            nn.Conv2d(13, 10, 3, padding=6,dilation=6),
            nn.SELU()
            )
        self.conv_d_12 = nn.Sequential(
            nn.BatchNorm2d(23),
            nn.Conv2d(23, 10, 3, padding=12,dilation=12),
            nn.SELU()
            )
        self.conv_d_18 = nn.Sequential(
            nn.BatchNorm2d(33),
            nn.Conv2d(33, 10, 3, padding=(18,18),dilation=(18,18)),
            nn.SELU()
            )
        self.conv_d_24 = nn.Sequential(
            nn.BatchNorm2d(43),
            nn.Conv2d(43, 10, 3, padding=(24,24),dilation=(24,24)),
            nn.SELU()
            )

        self.pool_bottom_pitch = nn.AvgPool2d((400,1))
        self.bottom_pitch = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.SELU()
            )

        self.up_conv3_pitch = nn.Sequential(
            nn.BatchNorm2d(77),
            nn.Conv2d(77, 64, 3, padding=1),
            nn.SELU()
            )

        self.up_conv2_pitch = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SELU()
            )

        self.up_conv1_pitch = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.SELU()
            )

        self.softmax_pitch = nn.Softmax(dim=2)

    def forward(self, x):

        conv1 = self.conv_d_3(x)        
        conv1_cat = torch.cat((x, conv1), dim=1) 

        conv2 = self.conv_d_6(conv1_cat) 
        conv2_cat = torch.cat((conv1_cat, conv2), dim=1) 

        conv3 = self.conv_d_12(conv2_cat) 
        conv3_cat = torch.cat((conv2_cat, conv3), dim=1) 

        conv4 = self.conv_d_18(conv3_cat)
        conv4_cat = torch.cat((conv3_cat, conv4), dim=1) 

        conv5 = self.conv_d_24(conv4_cat)
        conv5_cat = torch.cat((conv4_cat, conv5), dim=1) 

        final_cat = torch.cat((conv5_cat, x), dim=1)

        for i in range(7):
            final_cat = torch.cat((final_cat, x), dim=1)

        u3_ = self.up_conv3_pitch(final_cat)
        u2_ = self.up_conv2_pitch(u3_)
        u1_ = self.up_conv1_pitch(u2_)

        bm_ = self.pool_bottom_pitch(self.bottom_pitch(u3_))
              
        output_pitch = self.softmax_pitch(torch.cat((bm_, u1_), dim=2))

        return output_pitch

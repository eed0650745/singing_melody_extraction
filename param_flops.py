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
from thop import profile


Net = Net()
Net.cuda()
Net.float()
Net.train()

flops1, params1 = profile(Net, input_size=(1,3, 256,256))
print(flops1)
print(params1)

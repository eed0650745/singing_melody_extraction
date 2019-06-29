import numpy as np
import wave as we
from MSnet.cfp import cfp_process
import os



rootdir = "./raw/"
npz_dir = './train_NPZ/'


list1 = os.listdir(rootdir) 
wav_list=[]
for name in list1:

	wav_list.append(name.split('.')[0])


for namelist in wav_list:
	print(namelist)

	filepath = rootdir + '/' + namelist + '.wav'


	data, CenFreq_useless, time_arr = cfp_process(filepath, model_type='vocal')


	np_save_name = npz_dir + namelist + ".npz"


	np.savez(np_save_name,data = data,CenFreq = CenFreq_useless,time_arr = time_arr)


import numpy as np
import os
import csv
import mir_eval

result_dir = "./train_RESULT/"

label_dir = './train_LABEL/'

npz_dir = './train_NPZ/'




list1 = os.listdir(npz_dir) 
npz_list=[]
for name in list1:
	npz_list.append(name.split('.')[0]+'.npz')
pv_list=[]
for name in list1:
	pv_list.append(name.split('.')[0] + '.pv')
print(pv_list)



for namelist in npz_list:
	print(namelist)
	A = np.load(npz_dir + namelist)
	data = A['data']
	time_arr = A['time_arr']

	ref_time, ref_freq = mir_eval.io.load_time_series(label_dir+(namelist.split('.')[0])+'.pv')
	resample_f, _  = mir_eval.melody.resample_melody_series(ref_time, ref_freq, ref_freq, time_arr, kind='linear')

	string1 = ""	

	for i in range(len(time_arr)):
			string1 = string1 + str(("%.3f" % time_arr[i])) + '\t' +str(("%.3f" %resample_f[i])) + '\n'
			
	f1 = open(result_dir+namelist.split('.')[0]+"REF.txt",'w')
	f1.write(str(string1))
	f1.close()
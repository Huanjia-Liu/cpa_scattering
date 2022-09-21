import sys
import numpy as np 
local_path = "/data/SCA_code/Huanjia"
sys.path.append(local_path)
#from cap_scattering import cap_scattering
print(sys.path)
#from cap_scattering import cap_scattering
from cap_scattering import cpa_scattering


import h5py
data_path = "/data/SCA_data/ascad.sx"
in_file = h5py.File(data_path,'r')

in_file_pt = in_file['metadata/plain_text']
in_file_trace = in_file['traces']

in_file_pt = np.array(in_file_pt[:1000])
in_file_trace = np.array(in_file_trace[:1000])



mine_test = cpa_scattering(trace = in_file_trace,plant_text = in_file_pt,mode="np")
result = mine_test.run([0,1])
mine_test.show_result(result)


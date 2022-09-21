# -*- coding:utf-8 -*-
# author: Huanjia
# version: 0.0.3
# latest updated time: Sep, 15, 2022
# =================================================
#                      _oo0oo_
#                     o8888888o
#                     88" . "88
#                     (| -_- |)
#                     0\  =  /0
#                   ___/`---'\___
#                 .' \\|     |// '.
#                / \\|||  :  |||// \
#               / _||||| -:- |||||- \
#              |   | \\\  - /// |   |
#              | \_|  ''\---/''  |_/ |
#              \  .-\__  '-'  ___/-. /
#            ___'. .'  /--.--\  `. .'___
#         ."" '<  `.___\_<|>_/___.' >' "".
#        | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#        \  \ `_.   \_ __\ /__ _/   .-` /  /
#    =====`-.____`.___ \_____/___.-`___.-'=====
#                      `=---='
 
 
#    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#        Blessed By Buddha       No Bug Forever
# =================================================




import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D

# =================================================
# The class cpa_scattering combines the basic cpa algorithm and 
# scattering algorithm toghter. GPU supported matrix library Cupy 
# is also be added.
# =================================================
class cpa_scattering:

   # three arguments here, trace, plant_text should be in numpy or cupy format. 
   # Augument "mode" controls if GPU cupy is used, defualt value is "cp"  
   def __init__(self, trace, plant_text, mode = "cp"):
      try:
        trace.shape[-1]
        self.trace = trace
      except TypeError:
        print("The format of trace shoud support function shape")
      try:
        plant_text.shape[-1]
        self.plant_text = plant_text
      except TypeError:
        print("The format of plain text shoud support function shape")
      
      """string check should be added here!!!!!!!""" 
      self.mode = mode
      

   #sbox matrix
   np_sbox = np.array([
      0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
      
      0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
      0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
      0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
      0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
      0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
      0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
      0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
      0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
      0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
      0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
      0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
      0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
      0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
      0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
      0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
   ])

   #hamming weight
   np_hw = np.array([
      0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
   ])
#============================================================
# Cupy version

   #Function: cpa_cp (private function)
   #    Implementing cpa algorithm. 
   #    Input note:
   #        Byte_list is a 1D array or list controlling the index 
   #        of byte wanted to be decrpyted. exp: cp.array([1,14])
   #        means decrypt the second and the fifteenth bytes. 
   #    Output note:
   #        The output variable final result is a 3D cupy matrix. [bytes, key, time]
   def cpa_cp(self, plant_text, trace, bytes_list = np.arange(16)):
      
      key_guess = cp.arange(256)
      final_result = cp.zeros((len(bytes_list),256,trace.shape[-1]))

      for index in bytes_list:
         key_guess = cp.tile(cp.arange(256),(plant_text.shape[0],1))
         cp_sbox = cp.array(cpa_scattering.np_sbox)
         
         #target
         sbox_value = cp_sbox[cp.bitwise_xor(plant_text[:,index],key_guess.T).T]
         cp_hw = cp.array(cpa_scattering.np_hw)
         power_model = cp_hw[sbox_value]

         #correlation processing
         power_mean = cp.mean(power_model,axis=0)
         trace_mean = cp.mean(trace, axis=0)
         numerator = cp.matmul((power_model-power_mean).T,(trace-trace_mean))
         temp_denominator_power = cp.sum(cp.power(power_model-power_mean,2),axis=0).reshape((256,1))
         temp_denominator_trace = cp.sum(cp.power(trace-trace_mean,2),axis=0).reshape(1,trace.shape[-1])
         
         denomiator =cp.sqrt(cp.matmul(temp_denominator_power,temp_denominator_trace))
         correlation = numerator/denomiator

         final_result[index,:,:] = correlation
      return final_result
   #Function: cap_result_analysis_cp (private function)
   #    Analysis cpa results
   #    Input note:
   #        final_result is a 3D cupy matrix. [bytes, key, time], from function cpa_cp
   #    Output note:
   #        The output reulst_list is a list of dictionaries. Dictionary records three curve: key,
   #        lower bond, and upper bond as well as the key value who has the maxium correlations.
   def cap_result_analysis_cp(self,final_result):
      no_byte = final_result.shape[0]
      result_list = []

      #number of byte, then get upper bond, lower bond and key curve here
      for i in range(no_byte):
         #final_result = np.array(final_result.get())
         final_result_byte = final_result[i,:,:]
         #find the index of the row of the largest correlation 
         key = int(cp.unravel_index(final_result_byte.argmax(),final_result_byte.shape)[0])
         key_curve = final_result_byte[key]
         upper_bond_curve = final_result_byte.max(axis=0)
         lower_bond_curve = final_result_byte.min(axis=0)
         key_correlation = final_result_byte.max(axis=1)
         result_dic = {
            "key_curve": key_curve,
            "lower_bond_curve": lower_bond_curve,
            "upper_bond_curve": upper_bond_curve,
            "key_correlation": key_correlation
            }
         result_list.append(result_dic)
      return result_list

#============================================================
# Numpy version

   #Function: cap_np (private function)

   def cpa_np(self, plant_text, trace, bytes_list = np.arange(16)):
      key_guess = np.arange(256)
      final_result = np.zeros((len(bytes_list),256,trace.shape[-1]))

      for index in bytes_list:
         key_guess = np.tile(np.arange(256),(plant_text.shape[0],1))
         
         #target
         sbox_value = cpa_scattering.np_sbox[np.bitwise_xor(plant_text[:,index],key_guess.T).T]
         power_model = cpa_scattering.np_hw[sbox_value]

         #correlation processing
         power_mean = np.mean(power_model,axis=0)
         trace_mean = np.mean(trace, axis=0)
         numerator = np.matmul((power_model-power_mean).T,(trace-trace_mean))
         temp_denominator_power = np.sum(np.power(power_model-power_mean,2),axis=0).reshape((256,1))
         temp_denominator_trace = np.sum(np.power(trace-trace_mean,2),axis=0).reshape(1,trace.shape[-1])
         
         denomiator =np.sqrt(np.matmul(temp_denominator_power,temp_denominator_trace))
         correlation = numerator/denomiator

         final_result[index,:,:] = correlation
      return final_result

   #Function: cap_result_analysis_np (private function)

   def cap_result_analysis_np(self,final_result):
      no_byte = final_result.shape[0]
      result_list = []

      #number of byte, then get upper bond, lower bond and key curve here
      for i in range(no_byte):
         #final_result = np.array(final_result.get())
         final_result_byte = final_result[i,:,:]
         #find the index of the row of the largest correlation 
         key = int(np.unravel_index(final_result_byte.argmax(),final_result_byte.shape)[0])
         key_curve = final_result_byte[key]
         upper_bond_curve = final_result_byte.max(axis=0)
         lower_bond_curve = final_result_byte.min(axis=0)
         key_correlation = final_result_byte.max(axis=1)
         result_dic = {
            "key_curve": key_curve,
            "lower_bond_curve": lower_bond_curve,
            "upper_bond_curve": upper_bond_curve,
            "key_correlation": key_correlation
            }
         result_list.append(result_dic)
      return result_list


   #Function: show_result (private function)
   #    Plot the sample vs corrleation and print key
   #    Input note:
   #        result_list is a list of dictionaries from cap_result_analysis_np or cap_result_analysis_cp
   def show_result(self, result_list):
      count = 0
      for result_dic in result_list:
         count+=1
         plt.figure(figsize=(12,4))
         plt.subplot(1,len(result_list),count)
         plt.plot(cp.asnumpy(result_dic["lower_bond_curve"]), color = 'g')
         plt.plot(cp.asnumpy(result_dic["upper_bond_curve"]), color = 'b')
         plt.plot(cp.asnumpy(result_dic["key_curve"]),color='r')
         print(np.argmax(result_dic["key_correlation"]))
         plt.show()

   #Function: scattering (public function)
   #    Implementing scattering wavelet transform on traces
   #    Input note:
   """        J, Q  will be explain more detail in the future !!!!!!!!!!!!"""
   #    Output note:
   #        new trace
   def scattering(traces, J, Q):
      traces = np.array(traces.get())
      flag_for_np = False
      for trace in traces:
         T = trace.shape[-1]
         J = 8
         Q = 16
         scattering = Scattering1D(J, T, Q)
         Sx = scattering(trace)
         meta = scattering.meta()
         order0 = np.where(meta['order'] == 0)
         order1 = np.where(meta['order'] == 1)
         order2 = np.where(meta['order'] == 2)
         final_trace = Sx[order0][0]
         
         #Connecting all S outputs.
         for i in Sx[order2]:
            final_trace+= i
         for i in Sx[order2]:
            final_trace+=i
         if(flag_for_np == False):
            trace_scattering = np.empty((0,len(final_trace)))
            trace_scattering = np.vstack((trace_scattering,final_trace))
            flag_for_np = True
         else:
            trace_scattering = np.vstack((trace_scattering,final_trace))

      return trace_scattering


   #Function: run (private function)
   #    Running decryption
   #    Input note:
   #        Byte_list is a 1D array or list controlling the index 
   #        of byte wanted to be decrpyted. exp: cp.array([1,14]) or [1,14]
   #        means decrypt the second and the fifteenth bytes. 

   def run(self,byte_target):
      if self.mode == "cp":
         final_result = self.cpa_cp(self.plant_text,self.trace,byte_target)
         result_list = self.cap_result_analysis_cp(final_result)
      elif self.mode == "np":
         final_result = self.cpa_np(self.plant_text,self.trace,byte_target)
         result_list = self.cap_result_analysis_np(final_result)
      return result_list



#==================================
#example cpa in np

#in_file_pt = in_file['metadata/plain_text']
#in_file_trace = in_file['traces']
#in_file_pt = np.array(in_file_pt[:1000])
#in_file_trace = np.array(in_file_trace[:1000])
#test = cpa_scattering(trace = in_file_trace, plant_text= in_file_pt, mode="np")
#result = test.run([0,1])
#test.show_result(result)

#==================================
#example cpa in cp and scattering

#in_file_pt = in_file['metadata/plain_text']
#in_file_trace = in_file['traces']
#in_file_pt = cp.array(in_file_pt[:100])
#in_file_trace = cp.array(in_file_trace[:100,:])

#trace_scattering = cpa_scattering.scattering(in_file_trace,6,8)

#test = cpa_scattering(trace = trace_scattering, plant_text= in_file_pt, mode="cp")
#result = test.run([0,1])
#test.show_result(result)

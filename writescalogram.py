'''

Change recordings to scalograms
Store the results as .jpg

'''


import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import xlrd    
from PIL import Image
import scaleogram as scg
from numpy import savetxt



# %%Read Training data

#Demo of one data file
input_temp = loadmat('507_Depression_REST.mat',squeeze_me=True)
input_data = input_temp['EEG']
data_temp = input_data['data'][()][:,1000:21000]
[chan,steps] = data_temp.shape


#%%
def read_data(filename):
    number = []
    with open(filename) as f_input:
        for raw_row in f_input:
            values = raw_row.strip()
            number.append(values)
    #        print(values)
    
    sample = len(number)
    input = np.zeros((chan,20000,sample))
       
    iter = 0
         
    for i in number:
        print(i+'_Depression_REST.mat')
        x = '../Thesis/my_Thesis/data_mat/'+i+'_Depression_REST.mat'
        input_temp = loadmat(x,squeeze_me=True)
        data_in = input_temp['EEG']
        input[:,:,iter] = data_in['data'][()][0:66,1000:21000]
        iter += 1
        
    
    ###Read Training data label  
    iter = 0
    label = np.zeros(sample)
    for i in number:
        for sh in xlrd.open_workbook('../Thesis/my_Thesis/data_mat/Data_4_Import_REST.xlsx').sheets():  
            for row in range(sh.nrows):
                for col in range(sh.ncols):
                    myCell = sh.cell(row, col)
                    if myCell.value == int(i):
    #                    print('-----------')
    #                    print('Found!')
                        labelCell = sh.cell(row, col+1)
                        label[iter] = labelCell.value
        iter +=1                    
    
    label[label<3] = 1 #MDD
    label[label>10] = 0 #healthy
    
    return input, label


train, labeltrain = read_data('../Thesis/my_Thesis/data_mat/training.txt')
test,labeltest = read_data('../Thesis/my_Thesis/data_mat/testing.txt')
savetxt('labeltrain.csv', labeltrain, delimiter=',')
savetxt('labeltest.csv', labeltest, delimiter=',')
# %%
# Choose default wavelet function 
    
scg.set_default_wavelet('morl')
# print("Default wavelet function used to compute the transform:", scg.get_default_wavelet(), "(",
#       pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")


# range of scales to perform the transform
scales = scg.periods2scales( np.arange(1,1500) )

# Choose a channel
c = 30 #channel number


# Plot all training scalograms in Train_scalo folder
for i in range(len(labeltrain)):
    print(i)
    x_values_wvt_arr = range(0,len(train[c,:,i]),1)
    
    # the scaleogram
    scalo = scg.cws(train[c,:,i], scales=scales, figsize=(10, 4.0), coi = False, ylabel="Period", xlabel="Timestep",
            title=' '); 
    fig = scalo.figure
    fig.savefig('Train_scalo/'+str(i)+'.jpg')


# Plot all testing scalograms in Test_scalo folder
for ii in range(len(labeltest)):
    print(ii)
    x_values_wvt_arr = range(0,len(test[c,:,ii]),1)
    
    # the scaleogram
    scalo = scg.cws(train[c,:,ii], scales=scales, figsize=(10, 4.0), coi = False, ylabel="Period", xlabel="Timestep",
            title=' '); 
    fig = scalo.figure
    fig.savefig('Test_scalo/'+str(ii)+'.jpg')








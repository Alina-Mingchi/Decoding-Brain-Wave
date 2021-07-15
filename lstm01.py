'''

LSTM
PREDICT database
    
'''


import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import itertools
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from scipy.io import savemat
import xlrd

# %%Read Training data

#Demo of one data file
input_temp = loadmat('507_Depression_REST.mat',squeeze_me=True)
input_data = input_temp['EEG']
data_temp = input_data['data'][()][:,:]
[chan,steps] = data_temp.shape


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
        x = i+'_Depression_REST.mat'
        input_temp = loadmat(x,squeeze_me=True)
        data_in = input_temp['EEG']
        input[:,:,iter] = data_in['data'][()][0:66,1000:21000]
        iter += 1



###Read Training data label
iter = 0
    label = np.zeros(sample)
    for i in number:
        for sh in xlrd.open_workbook('Data_4_Import_REST.xlsx').sheets():
            for row in range(sh.nrows):
                for col in range(sh.ncols):
                    mycell = sh.cell(row, col)
                    if mycell.value == int(i):
                        #                    print('-----------')
                        #                    print('Found!')
                        labelcell = sh.cell(row, col+1)
                        label[iter] = labelcell.value
        iter +=1

        label[label<3] = 1 #MDD
        label[label>10] = 0 #healthy
    return input, label

train, labeltrain = read_data('training.txt')
test,labeltest = read_data('testing.txt')

savemat('train.mat', {'train':train})
savemat('labeltrain.mat', {'labeltrain':labeltrain})
savemat('test.mat', {'test':test})
savemat('labeltest.mat', {'labeltest':labeltest})

c = 30 #channel

#Prepare the data for 5 fold validation
# Uncomment if need a certain channel
t1 = train[c,:,0:19]
t2 = train[c,:,19:38]
t3 = train[c,:,38:57]
t4 = train[c,:,57:76]
t5 = train[c,:,76:95]

# Uncomment if need full channel data
# t1 = train[:,:,0:19]
# t2 = train[:,:,19:38]
# t3 = train[:,:,38:57]
# t4 = train[:,:,57:76]
# t5 = train[:,:,76:95]

lt1 = labeltrain[0:19]
lt2 = labeltrain[19:38]
lt3 = labeltrain[38:57]
lt4 = labeltrain[57:76]
lt5 = labeltrain[76:95]


# %%
end_point = 10
input_size = 20000
hidden_size = 50
batch_size = 1
n_class = 2
drop = 0.2
num_layers = 2

# Uncomment if trying to find optimal number of layers
# layerarr = np.array([1,2,3,4])
# for num in range(4):
#     num_layers = layerarr[num]
#     print('Number of layer is ',num_layers)

# Uncomment if trying to find optimal dropout rate
droparr = np.array([0.1,0.2,0.5])
for count in range(3):
    drop = droparr[count]
    print('Dropout rate is', drop)
    
    def ToVariable(x):
        tmp = torch.FloatTensor(x)
        return Variable(tmp)
    
    #Build the LSTM model for classification
    class LSTMcla(torch.nn.Module):
        
        def __init__(self,input_size,hidden_dim,n_class,batch_size,drop,num_layers):
            super(LSTMcla,self).__init__()
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.n_class = n_class
            self.batch_size = batch_size
            self.drop = drop
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(input_size,hidden_dim,num_layers)
            self.drop_layer = torch.nn.Dropout(drop)
            self.linear_layer = torch.nn.Linear(batch_size * hidden_dim,n_class)

            self.hidden = self.init_hidden()
        
        
        def init_hidden(self):
            return (Variable(torch.rand(num_layers,self.batch_size,self.hidden_dim)),
                    Variable(torch.rand(num_layers,self.batch_size,self.hidden_dim)))
        
        def forward(self,seq):
            lstm_out, self.hidden = self.lstm(seq.view(len(seq),self.batch_size,-1),self.hidden)
            output = self.drop_layer(lstm_out[:,-1])
            output = self.linear_layer(output)
            
            return output
    
    
    
    model = LSTMcla(input_size,hidden_size,n_class,batch_size,drop,num_layers)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = 0.015,momentum = 0.8)
    
    
    # %Separate data to 5 folds
    # Uncomment for single channel processing
    seq1 = np.concatenate((t1,t2,t3,t4),axis = 1)
    label1 = np.concatenate((lt1,lt2,lt3,lt4))
    val1 = t5
    labelval1 = lt5
    
    seq2 = np.concatenate((t1,t2,t3,t5),axis = 1)
    label2 = np.concatenate((lt1,lt2,lt3,lt5))
    val2 = t4
    labelval2 = lt4
    
    seq3 = np.concatenate((t1,t2,t5,t4),axis = 1)
    label3 = np.concatenate((lt1,lt2,lt5,lt4))
    val3 = t3
    labelval3 = lt3
    
    seq4 = np.concatenate((t1,t5,t3,t4),axis = 1)
    label4 = np.concatenate((lt1,lt5,lt3,lt4))
    val4 = t2
    labelval4 = lt2
    
    seq5 = np.concatenate((t5,t2,t3,t4),axis = 1)
    label5 = np.concatenate((lt5,lt2,lt3,lt4))
    val5 = t1
    labelval5 = lt1
    
    
    
    # Uncomment for full channel processing
    # seq1 = np.concatenate((t1,t2,t3,t4),axis = 2)
    # label1 = np.concatenate((lt1,lt2,lt3,lt4))
    # val1 = t5[1,:,:]
    # labelval1 = lt5
    
    # seq2 = np.concatenate((t1,t2,t3,t5),axis = 2)
    # label2 = np.concatenate((lt1,lt2,lt3,lt5))
    # val2 = t4[1,:,:]
    # labelval2 = lt4
    
    # seq3 = np.concatenate((t1,t2,t5,t4),axis = 2)
    # label3 = np.concatenate((lt1,lt2,lt5,lt4))
    # val3 = t3[1,:,:]
    # labelval3 = lt3
    
    # seq4 = np.concatenate((t1,t5,t3,t4),axis = 2)
    # label4 = np.concatenate((lt1,lt5,lt3,lt4))
    # val4 = t2[1,:,:]
    # labelval4 = lt2
    
    # seq5 = np.concatenate((t5,t2,t3,t4),axis = 2)
    # label5 = np.concatenate((lt5,lt2,lt3,lt4))
    # val5 = t1[1,:,:]
    # labelval5 = lt1
    
    
    # temporary storage of data to be passed to the model for each fold
    seqarr = np.array([seq1,seq2,seq3,seq4,seq5])
    labelarr = np.array([label1,label2,label3,label4,label5])
    valarr = np.array([val1,val2,val3,val4,val5])
    labelvalarr = np.array([labelval1,labelval2,labelval3,labelval4,labelval5])
    
    tlossarr = np.zeros((5,200))
    vlossarr = np.zeros((5,200))
    

    
    for k in range(5):
        print('Fold')
        print(k)
        seq = seqarr[k,:,:]
        label = labelarr[k,:]
        val = valarr[k,:,:]
        labelval = labelvalarr[k,:]
        
        
        seq = seq.T
        seq = torch.Tensor(seq)
        label = torch.LongTensor(label)
        
        val = val.T
        val = torch.Tensor(val)
        labelval = torch.LongTensor(labelval1)
        
        
        with torch.no_grad():
            output = model(seq)
            out_s = F.softmax(output)
            out_arr = out_s.detach().numpy()
            predict = out_arr[:,0] > out_arr[:,1]
            predict = predict * 1
            predict = 1 - predict
        # print('Before Training')
        # print(predict)
        
        vloss = np.zeros((0))
        tloss = np.zeros((0))
        
        for epoch in range(200):
            # print(epoch)
            
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            output= model.train()(seq)
            valoutput = model.eval()(val)
            
            loss = loss_function(output,label)
            tloss = np.append(tloss,loss.detach().numpy())
            # print('Training')
            #    print(tloss)
            loss.backward()
            optimizer.step()
            
            out_s = F.softmax(valoutput)
            out_arr = out_s.detach().numpy()
            valpredict = out_arr[:,0] > out_arr[:,1]
            valpredict = valpredict * 1
            valpredict = 1 - valpredict
            #    print('Validation')
            #    print(valpredict)
            valloss = loss_function(valoutput,labelval)
            vloss = np.append(vloss,valloss.detach().numpy())
        #    print('Validation')
        #    print(vloss)
        
        with torch.no_grad():
            output = model(seq)
            out_s = F.softmax(output)
            out_arr = out_s.detach().numpy()
            predict = out_arr[:,0] > out_arr[:,1]
            predict = predict * 1
            predict = 1 - predict
            print('After Training')
            print(predict)

# Plot of training and validation loss
        plt.plot(tloss,label = "Training")
        plt.plot(vloss,label = "Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss over epoch')
        plt.rcParams.update({'font.size': 16})
        plt.legend()
        plt.show()
        
        tlossarr[k,:] = tloss
        vlossarr[k,:] = vloss
            

    avgtloss = np.mean(tlossarr,axis=0)
    avgvloss = np.mean(vlossarr,axis=0)
            
    plt.plot(avgtloss,label = 'Training Loss')
    plt.plot(avgvloss,label = 'Validation Loss')
    plt.rcParams.update({'font.size': 16})
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss over epoch')
    plt.legend()
    plt.show()
            
    print('Minimum of validation loss is: ',np.min(avgvloss))
    print('Optimal epoch is: ',np.argmin(avgvloss))


# %% Train the model for testing

seq = train[c,:,:]
label = labeltrain

seq = seq.T
seq = torch.Tensor(seq)
label = torch.LongTensor(label)

model = LSTMcla(input_size,hidden_size,n_class,batch_size,drop,num_layers)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.025,momentum = 0.8)


trloss = []

for epoch in range(np.argmin(avgvloss)+1):
    print(epoch)
    
    optimizer.zero_grad()
    model.hidden = model.init_hidden()
    output= model(seq)
    
    loss = loss_function(output,label)
    trloss = np.append(trloss,loss.detach().numpy())
    print('Training')
    print(trloss)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    output = model(seq)
    out_s = F.softmax(output)
    out_arr = out_s.detach().numpy()
    predict = out_arr[:,0] > out_arr[:,1]
    predict = predict * 1
    predict = 1 - predict
    print('After Training')
    print(predict)



plt.plot(trloss,label = "Training")
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Loss over epoch')
plt.legend()
plt.show()


# %% Define plotting of confusion_matrix (from sklearn package)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.rcParams.update({'font.size': 26})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% Testing
seqq = test[1,:,:]
seqq = seqq.T
seqq = torch.Tensor(seqq)
labeltest = torch.LongTensor(labeltest)

with torch.no_grad():
    testout = model.eval()(seqq)
    testout_s = F.softmax(testout)
    testout_arr = testout_s.detach().numpy()
    tpredict = testout_arr[:,0] > testout_arr[:,1]
    tpredict = tpredict * 1
    tpredict = 1 - tpredict
    print('Predict')
    print(tpredict)


cm = confusion_matrix(labeltest, tpredict)
cm
categories = ('Healthy','Patient')
plt.rcParams.update({'font.size': 26})
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, categories)


#############################Reference######################
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# https://docs.scipy.org/doc/scipy/reference/tutorial/io.html


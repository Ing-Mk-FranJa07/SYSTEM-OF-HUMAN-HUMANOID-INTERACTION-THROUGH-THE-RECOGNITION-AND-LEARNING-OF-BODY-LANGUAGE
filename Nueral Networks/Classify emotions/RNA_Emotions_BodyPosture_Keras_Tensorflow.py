'''
Created on 8/08/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This script build a neural network which classify the mood of one person in 
six categories: Happy, Sad, Angry, Surprised, Reflexive and Normal. 

The data used like input is a vector which length is 23 cells, each position has 
an angle that describe the posture of the body. 

The output of the neural network is a binary vector with 6 cells that is a 
specific code to represent each category described previously. 
'''
# Wrappers to handling of data.
import pandas as pd
import numpy as np

# Wrapper to build, train and test neural networks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Wrappers used to plot the confusion matrix.
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def Plot_Confusion_Matrix(Matrix, Title, Cmap, Normalize = 'True'):
    '''
    Function that allow plot the confusion matrix, in two formats, 
    percentage or total data.
    '''
    if Normalize:
        Confusion_Matrix = Matrix.astype('float') / Matrix.sum(axis=1)[:, np.newaxis]
    else:
        Confusion_Matrix = Matrix
        
    plt.imshow(Confusion_Matrix, interpolation = 'nearest', cmap = Cmap)
    plt.title(Title)
    plt.colorbar()
    tick_marks = np.arange(len(Class_Name))
    plt.xticks(tick_marks, Class_Name, rotation = 45)
    plt.yticks(tick_marks, Class_Name)
    
    fmt = '.2f' if Normalize else 'd' 
    thresh = Confusion_Matrix.max() / 2.
    
    for i, j in itertools.product(range(Confusion_Matrix.shape[0]), range(Confusion_Matrix.shape[1])):
        plt.text(j, i, format(Confusion_Matrix[i, j], fmt), horizontalalignment = "center", color = "white" if Confusion_Matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Desired Output')
    plt.xlabel('Estimated Output')
    
    plt.show()

# Are loaded the Train and Test data.
Data_Train = pd.read_csv("...\DataSet_Organized\DataSetPostures_Train.csv", header = 0, index_col = 0)
Data_Test = pd.read_csv("...\DataSet_Organized\DataSetPostures_Test.csv", header = 0, index_col = 0)

# Are created the Train and Test input and output.
X_Train = np.array(Data_Train.ix[:,2:25]) 
Y_Train = np.array(Data_Train.ix[:,25:33])
X_Test = np.array(Data_Test.ix[:,2:25])
Y_Test = np.array(Data_Test.ix[:,25:33])

# Are created vectors that contain the index of the each emotion class.
Index_Y_Test = np.empty([len(Y_Test),1], dtype = np.float64)
Index_Y_Pred = np.empty([len(Y_Test),1], dtype = np.float64)

# Are defined the parameters of the neural network and the model.
HL_1_Nodes = 23
HL_2_Nodes = 17
RNA = Sequential()
 
# Is created the first hidden layer (Activation function = linear rectified, Nodes = 23) 
RNA.add(Dense(HL_1_Nodes, activation = 'relu', input_shape = (X_Train.shape[1],))) 
RNA.add(Dropout(0.05)) # Function that prevent the over training.
 
# Is created the second hidden layer (Activation function = linear rectified, Nodes = 17)
RNA.add(Dense(HL_2_Nodes, activation = 'relu')) 

# Is created the output layer (Activation function = Sigmoidal, Nodes = 6)
RNA.add(Dense(6, activation = 'sigmoid')) 

# Is defined the train parameters of the neural network.
sgd = SGD(lr = 0.025, decay = 1e-5, momentum = 0.15, nesterov = False)
RNA.compile(loss = 'categorical_crossentropy', # Optimization function.
            optimizer = sgd,                   # Optimization type: Stochastic downward gradient
            metrics = ['accuracy'])
 
# Is trained the neural network.
Training = RNA.fit(X_Train, Y_Train, epochs = 25, batch_size = 1, verbose = 2); print('\n'*4)

# Is tested the neural network.
Y_Pred = RNA.predict(X_Test); print('Prediction of the output of the test data','\n',Y_Pred); print('\n'*4)
 
# Is evaluated the perform of the neural network.
Score = RNA.evaluate(X_Test, Y_Test, verbose = 2) 
print('Accuracy of the neural network', '\n', Score)

# Is determined the index of the each class of the output test and output predicted.
for i in range(len(Y_Test)):
    Index_Y_Test[i] = np.argmax(Y_Test[i,:]) 
    Index_Y_Pred[i] = np.argmax(Y_Pred[i,:])

# Is calculated the confusion matrix.
Confusion_Matrix = confusion_matrix(Index_Y_Test,Index_Y_Pred)

# Are determined the set up to plot the confusion matrix.
Class_Name = ['Normal', 'Reflexive', 'Surprised', 'Angry', 'Sad', 'Happy']
Title = 'Confusion Matrix'
Cmap = plt.cm.Blues

# Are printed the confusion matrix using the both formats.
Plot_Confusion_Matrix(Confusion_Matrix, Title, Cmap, Normalize = True)
Plot_Confusion_Matrix(Confusion_Matrix, Title, Cmap, Normalize = False)

# Is saved the model of the neural network.
Version = '1' # Version of the model created.
RNA.save('...\Model_RNA_Recognition_Of_Emotions ' + Version)

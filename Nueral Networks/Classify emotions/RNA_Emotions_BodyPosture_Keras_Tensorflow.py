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

# Are loaded the Train and Test data.
Data_Train = pd.read_csv("...\DataSet_Organized\DataSetPostures_Train.csv", header = 0, index_col = 0)
Data_Test = pd.read_csv("...\DataSet_Organized\DataSetPostures_Test.csv", header = 0, index_col = 0)

# Are created the Train and Test input and output.
X_Train = np.array(Data_Train.ix[:,2:25]) 
Y_Train = np.array(Data_Train.ix[:,25:33])
X_Test = np.array(Data_Test.ix[:,2:25])
Y_Test = np.array(Data_Test.ix[:,25:33])

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
print('Accuracy of the neural network', '\n', Score); print('\n'*4)

# Is saved the model of the neural network.
Version = '1' # Version of the model created.
RNA.save('...\Model_RNA_Recognition_Of_Emotions ' + Version)

'''
Created on 31/08/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This script build a GAN neural network. Generative Adversarial Network.

The objective of this network is try to use the GAN to create new motion sequences
to increase the animation's data base used to be played by Robot Pepper. 

It has 32 motion sequences created previously, the idea is generated new animations 
similar to the originals, in that way the Generative network goes to create new
data from random numbers, and the Discriminator network goes to decide if the data
presented is real or created for the Generator network, first is trained the 
Discriminator network, and then the both networks are trained to increase the 
perform of the Generative network. 

The train parameters of the networks can be modified from the main of the script
in the line that call the train function.
'''
import time

# Wrappers to handling of data.
import pandas as pd
import numpy as np

# Wrapper to build, train and test neural networks
from keras.layers import LeakyReLU, Dropout
from keras.layers import Reshape, Dense, Activation, Flatten
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD

# Wrapper to plot data.
import matplotlib.pyplot as plt

class TakeTime(object):
    '''
    Wrapper object that determined the time employee to train the neural network.
    '''
    def __init__(self):
        '''
        Function that initializes the counting.
        '''
        self.StarTime = time.time()                                             # Variable that get the starting time.
    
    def Time(self):
        '''
        Function that determined how many seconds, minutes and hours have elapsed.
        '''
        sec = time.time() - self.StarTime                                       # Compute the time elapsed.
        
        if sec < 60:                                                            # Show the time elapsed in the format SS; MM; HH
            print("Time: " +  str(sec) + " seconds")
        elif sec < (60 * 60):
            print("Time: " + str(sec / 60) + " minutes")
        else:
            print("Time: " + str(sec / (60 * 60)) + " hours")

class DCGAN(object):
    '''
    Wrapper class that set up the input data, created the both networks 
    and train the system.
    '''
    def __init__(self):
        '''
        Function that set up the Data set.
        The original data is a 39*15 matrix, are added one raw and one column 
        to form a 40*16 matrix, to ease the manipulation of the networks.
        '''
        # Parameters used to build the input data set.
        Repetition = 1                                                          # Variable used to increase the original animation in the input data.
        plus = 0                                                                # Variable used to increase the original animation in the input data.
        self.DataSet = np.empty([Repetition*32,40,16])                          # Input Data.
        
        # Is loaded the motion sequences.
        for D in range(Repetition):
            # Motion sequences used to represent emotions:
            for i in range(1,11):                
                FileName = str("...\Motion_Sequences\ Emotion " + str(i) + ".csv")
                Data = pd.read_csv(FileName, header = 0, index_col = 0)

                self.DataSet[plus + i - 1, 0, :] = np.array(Data.ix[0,:])
                self.DataSet[plus + i - 1, 1:, :] = np.array(Data.ix[:,:])

    #             self.DataSet[i - 1, 0, :] = np.array(Data.ix[0,:])
    #             self.DataSet[i - 1, 1:, :] = np.array(Data.ix[:,:])
                
#                 for D in range(Repetition):#                      
#                     self.DataSet[plus + D, 0, :] = np.array(Data.ix[0,:])
#                     self.DataSet[plus + D, 1:, :] = np.array(Data.ix[:,:])
#              
#                 plus += Repetition
                
            # Motions sequences used to used in conversations.
            for i in range(1,23):                 
                FileName = str("...\Motion_Sequences\ Animation " + str(i) + ".csv")
                Data = pd.read_csv(FileName, header = 0, index_col = 0)

                self.DataSet[plus + 9 + i, 0, :] = np.array(Data.ix[0,:])
                self.DataSet[plus + 9 + i, 1:, :] = np.array(Data.ix[:,:])
    
    #             self.DataSet[9 + i, 0, :] = np.array(Data.ix[0,:])
    #             self.DataSet[9 + i, 1:, :] = np.array(Data.ix[:,:])
                
#                 for D in range(Repetition):#                   
#                     self.DataSet[plus + D, 0, :] = np.array(Data.ix[0,:])
#                     self.DataSet[plus + D, 1:, :] = np.array(Data.ix[:,:])
#                  
#                 plus += Repetition
        
#            plus += 31
            
        # Set up the input data.
        self.DataSet = self.DataSet.reshape(self.DataSet.shape[0], self.DataSet.shape[1], self.DataSet.shape[2], 1)
        self.DataSet = self.DataSet.astype('float32')
        
        # Is calculated the maximum value of the input data.
        self.DataMax = np.max(self.DataSet)
        
        # Normalize the input data; -1.0 <= input data <= 1.0.
        self.DataSet = self.DataSet / self.DataMax
        
        print('Data set structure:',' \n'*2, self.DataSet.shape, '\n'*2,
              'Data set range:', '\n'*2, np.min(self.DataSet), np.max(self.DataSet), '\n'*2)

        # Are defined the neural network parameters.
            # Neural Networks input.
        self.Input_Gen = 100                                                    # Shape of the Generator input
        self.Input_Dis = (self.DataSet.shape[1:])                               # Shape of the Discriminator input = [Rows, Cols, Chan].
            # Hidden layers outputs.
        self.Depth_Dis = 64                                                     # Initial depth of the Discriminator layers.                      
        self.Depth = 256                                                        # Initial depth of the Generator layers.
        self.Width = 5                                                          # Initial width of the Generator network.
        self.Length = 2                                                         # Initial length of the Generator network.
            # Convolutionals layers parameters.
        self.Kernel = 4
        self.strides = 2
            # Neural networks train parameters.
        self.Dropout_rate = 0.1                                                 # Percentage to dropout the input data.
        self.Momentum = 0.2                                                         
        self.Momentum_D = 0.2
        self.Momentum_A = 0.05
        LearningRate_D = 2e-2                                                   # Discriminator network learning rate.   
        Decrement_D = 1e-5                                                      # Discriminator network decay.
        LearningRate_A = 25e-2                                                  # Adversarial network learning rate.
        Decrement_A = 1e-5                                                      # Adversarial network decay.
            # Neural networks optimizers .
                # Adversarial network.
        self.Adam_A = Adam(lr = 1e-4) 
        self.RMS_A = RMSprop(lr = LearningRate_A, decay = Decrement_A)
        self.SGD_A = SGD(lr = LearningRate_A, decay = Decrement_A, momentum = self.Momentum_A, nesterov = False)
                # Discriminator network.
        self.Adam_D = Adam(lr = 1e-4) 
        self.RMS_D = RMSprop(lr = LearningRate_D, decay = Decrement_D)
        self.SGD_D = SGD(lr = LearningRate_D, decay = Decrement_D, momentum = self.Momentum_D, nesterov = False)
        
    def ShuffleDataSet(self, shuffle = False):
        '''
        Function that shuffle the input data, and create a validation data.
        *** Was not used.
        '''
        if shuffle:
            X_Train = self.DataSet[0:int(len(self.DataSet)*0.85), :, :, :]
            X_Test= self.DataSet[int(len(self.DataSet)*0.85):, :, :, :]
        
            return X_Train, X_Test
        else:
            return self.DataSet, np.zeros([1, 40, 16, 1])
  
    def Plot_Loss(self, Losses):
        '''
        Function that plot the Adversarial and Discriminative Losses.
        '''
        plt.figure(figsize=(10,8))
        plt.plot(Losses['D'], label = 'Discriminative loss')
        plt.plot(Losses['A'], label = 'Adversarial loss')
        plt.legend()
        plt.show()

    def Plot_Acc(self, Acc):
        '''
        Function that plot the Adversarial and Discriminative accuracies.
        '''
        plt.figure(figsize=(10,8))
        plt.plot(Acc['D'], label = 'Discriminative accuracy')
        plt.plot(Acc['A'], label = 'Adversarial accuracy')
        plt.legend()
        plt.show()
        
    def CreateGen(self):
        '''
        Function that create the model of the Generator network.
        '''
        # Is created the Generator network model.
        Gen = Sequential()
        
        # First layer of the network.
        Gen.add(Dense(self.Width* self.Length* self.Depth, input_dim = self.Input_Gen))
        Gen.add(BatchNormalization(momentum = self.Momentum))
        Gen.add(Activation('tanh'))
        Gen.add(Reshape((self.Width, self.Length, self.Depth))) 
        Gen.add(Dropout(self.Dropout_rate))
        
        # Second layer of the network.
        Gen.add(UpSampling2D(size = (2, 2)))
        Gen.add(Conv2DTranspose(int(self.Depth/2), self.Kernel, border_mode = 'same'))
        Gen.add(BatchNormalization(momentum = self.Momentum))
        Gen.add(Activation('tanh'))
        
        # Third layer of the network.
        Gen.add(UpSampling2D(size = (2, 2)))
        Gen.add(Conv2DTranspose(int(self.Depth/4), self.Kernel, border_mode = 'same'))
        Gen.add(BatchNormalization(momentum = self.Momentum))
        Gen.add(Activation('tanh'))
        
        # Fourth layer of the network.
        Gen.add(UpSampling2D(size = (2, 2)))
        Gen.add(Conv2DTranspose(int(self.Depth/8), self.Kernel, border_mode = 'same'))
        Gen.add(BatchNormalization(momentum = self.Momentum))
        Gen.add(Activation('tanh'))

        # Fifth layer of the network.
        Gen.add(Conv2DTranspose(int(self.Depth/16), self.Kernel, border_mode = 'same'))
        Gen.add(BatchNormalization(momentum = self.Momentum))
        Gen.add(Activation('tanh'))
         
        # Output layer.
        Gen.add(Conv2DTranspose(1, self.Kernel, border_mode = 'same'))
        Gen.add(Activation('tanh'))
        
        return Gen

    def CreateDis(self):
        '''
        Function that create the model of the Discriminator network.
        '''
        # Is created the Discriminator network model.
        Dis = Sequential()
        
        # First layer of the network.
        Dis.add(Conv2D(self.Depth_Dis, self.Kernel, strides = self.strides, input_shape = self.Input_Dis, padding = 'same'))
        Dis.add(LeakyReLU(alpha = 0.2))
        Dis.add(Dropout(self.Dropout_rate))
        
        # Second layer of the network.
        Dis.add(Conv2D(self.Depth_Dis*2, self.Kernel, strides = self.strides, padding = 'same'))
        Dis.add(LeakyReLU(alpha = 0.2))
        Dis.add(Dropout(self.Dropout_rate))
        
        # Third layer of the network.
        Dis.add(Conv2D(self.Depth_Dis*4, self.Kernel, strides = self.strides, padding = 'same'))
        Dis.add(LeakyReLU(alpha = 0.2))
        Dis.add(Dropout(self.Dropout_rate))
         
        # Fourth layer of the network.
        Dis.add(Conv2D(self.Depth_Dis*8, self.Kernel, strides = int(self.strides/self.strides), padding = 'same'))
        Dis.add(LeakyReLU(alpha = 0.2))
        Dis.add(Dropout(self.Dropout_rate))

        # Fifth layer of the network.
        Dis.add(Conv2D(self.Depth_Dis*16, self.Kernel, strides = int(self.strides/self.strides), padding = 'same'))
        Dis.add(LeakyReLU(alpha = 0.2))
        Dis.add(Dropout(self.Dropout_rate))

        # Output layer.
        Dis.add(Flatten())
        Dis.add(Dense(1))
        Dis.add(Activation('sigmoid'))
        
        return Dis
    
    def DiscriminatorModel(self, OptimizerType = 0):
        '''
        Function that implement the Discriminator model.
        '''
        # Model used to saved the Discriminator network.
        D = Sequential()
        
        # Is added the Discriminator network.
        D.add(self.CreateDis())

        # Is definite the network training.
        if OptimizerType == 0:
            D.compile(loss = 'binary_crossentropy', optimizer = self.Adam_D, metrics = ['accuracy'])
        elif OptimizerType == 1:
            D.compile(loss = 'binary_crossentropy', optimizer = self.RMS_D, metrics = ['accuracy'])
        elif OptimizerType == 2:
            D.compile(loss = 'binary_crossentropy', optimizer = self.SGD_D, metrics = ['accuracy'])
        
        return D
    
    def AdversarialModel(self, OptimizerType = 0):
        '''
        Function that implement the Adversarial network (Generator + Discriminator)
        '''
        # Is created a model to saved the both netwroks.
        A = Sequential()
        
        # Is added the Generator network. 
        A.add(self.CreateGen())
        
        # Is added the Discriminator network.
        A.add(self.CreateDis())
        
        # Is definet the network training.
        if OptimizerType == 0:
            A.compile(loss = 'binary_crossentropy', optimizer = self.Adam_A, metrics = ['accuracy'])
        elif OptimizerType == 1:
            A.compile(loss = 'binary_crossentropy', optimizer = self.RMS_A, metrics = ['accuracy'])
        elif OptimizerType == 2:
            A.compile(loss = 'binary_crossentropy', optimizer = self.SGD_A, metrics = ['accuracy'])
        
        return A

    # Se crea una funcion que permite entrenar el modelo de la red GAN.
    def Train(self, Iterations = 10, Batch_Size = 32, Version = str('1'), shuffle = False, OptimizerType = 0, Plot_Freq = 300):
        '''
        Function that train the GAN model.
        First is trained the Discriminator network, using the original motions 
        sequences and motion sequences created by the Generator network without 
        be trained; then is trained the Adversarial network (Gen + Dis) to 
        increase the performer of the Generator network.
        '''
        X_Train, X_Test = self.ShuffleDataSet(shuffle)
        
        # Are created the models of the neural network.
        Adversarial = self.AdversarialModel(OptimizerType)
        Discriminator = self.DiscriminatorModel(OptimizerType)
        Generator = self.CreateGen()
        
        ListAnimations = np.empty([Iterations, Batch_Size, 39, 16])             # List used to save the new motion sequences.
    
        Losses = {'D':[], 'A':[]}                                               # List used to save the discriminative and the adversarial losses.
        Accs = {'D':[], 'A':[]}                                                 # List used to save the discriminative and the adversarial accuracies.
        
        # The networks are trained.
        for I in range(Iterations):
            Index = np.random.randint(X_Train.shape[0], size = Batch_Size)      # List used to saved the index of the motion sequence selected.
            
            # Are loaded the animations in a random order.
            Animations = X_Train[Index, :, :, :]

            # Is created the input of the Generator model.
            Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
            
            # Are created new motion sequences.
            FakeAnimations = Generator.predict(Noise)
            
            # Is created the training set to the Discriminator network.
            X = np.concatenate((Animations, FakeAnimations))
            Y = np.ones([2*Batch_Size, 1]); Y[Batch_Size:, :] = 0
            
            # The Discriminator network is trained 
            Dis_loss = Discriminator.train_on_batch(X, Y)
            
            # Is created the training set to the Adversarial network.
            Adv_Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
            Adv_Y = np.ones([Batch_Size, 1])
            
            # The Adversarial network is trained.
            Adv_loss = Adversarial.train_on_batch(Adv_Noise, Adv_Y)
            
            # Is showed the evolution of the training.
            log_mesg = "%d: [D loss: %f, acc: %f]" % (I, Dis_loss[0], Dis_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, Adv_loss[0], Adv_loss[1])
            print(log_mesg)
         
            # Is saved the discriminative and the adversarial losses to be plotted.
            Losses['D'].append(Dis_loss[0])
            Losses['A'].append(Adv_loss[0])
            
            # Is saved the discriminative and the adversarial accuracies to be plotted.
            Accs['D'].append(Dis_loss[1])
            Accs['A'].append(Adv_loss[1])
            
            # Is denormalize the new motion sequences.
            FakeAnimations = FakeAnimations.reshape(FakeAnimations.shape[0], FakeAnimations.shape[1], FakeAnimations.shape[2])
            FakeAnimations = FakeAnimations[:, 1: ,:]
            FakeAnimations = FakeAnimations * self.DataMax
            
            # The new motion sequences are added to the list.
            ListAnimations[I,:,:,:] = FakeAnimations
           
            # Are plotted the both losses and accuracies.
            if I % Plot_Freq == Plot_Freq - 1:
                self.Plot_Loss(Losses)
                self.Plot_Acc(Accs)
        
        # Are saved the nueral networks models.
        Adversarial.save('...\Adversarial Model ' + Version)
        Discriminator.save('...\Discriminator Model ' + Version)
        Generator.save('...\Generator Model ' + Version)
 
        # Are saved the new motion sequences in .cvs files.
        for L in range(len(ListAnimations)):
            for A in range(len(FakeAnimations)) :
                DataFrame = pd.DataFrame(ListAnimations[L, A, :, :])
                DataFrame.to_csv("...\DataBaseGeneratedByRNA\ NewAnimation " + str(L+1) + "-" + str(A+1) + ".csv", sep = ",", header = False, index = False, index_label = 'Node')
        
if __name__ == '__main__':
    
    GAN = DCGAN()
    Timer = TakeTime()
    GAN.Train(Iterations = 300, Batch_Size = 32, Version = str('1'), shuffle = False, OptimizerType = 2, Plot_Freq = 300)
    Timer.Time()

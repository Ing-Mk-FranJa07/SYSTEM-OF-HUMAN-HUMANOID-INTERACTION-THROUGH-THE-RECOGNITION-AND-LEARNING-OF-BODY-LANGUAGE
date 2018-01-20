'''
Created on 31/08/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This script build a GAN neural network. Generative Adversarial Network.

The objective of this network is try to use the GAN to create new motion sequences
to increase the animation's data base used to be played by Robot Pepper. 

It has 32 motion sequences created previously, the idea is generated new animations 
similar to the originals, in that way the Generative network goes to create new
data from random numbers, and the Discriminative network goes to decide if the data
presented is real or created for the Generative network, first is trained the 
Discriminative network, and then the both networks are trained to increase the 
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
        The original data is a 39*16 matrix, are added one raw to form a 40*16 
        matrix, to ease the manipulation of the networks.
        '''
        # Parameters used to build the input data set.
        Repetition = 100                                                          # Variable used to increase the original animation in the input data.
        plus = 0                                                                # Variable used to increase the original animation in the input data.
        self.DataSet = np.empty([Repetition*32,40,16])                          # Input Data.
        
        # Is loaded the motion sequences.
        for D in range(Repetition):
            # Motion sequences used to represent emotions:
            for i in range(1,11):
                FileName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ Emotion " + str(i) + ".csv")
                Data = pd.read_csv(FileName, header = 0, index_col = 0)

                self.DataSet[plus + i - 1, 0, :] = np.array(Data.ix[0,:])
                self.DataSet[plus + i - 1, 1:, :] = np.array(Data.ix[:,:])
                
            # Motions sequences used to used in conversations.
            for i in range(1,23):
                FileName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ Animation " + str(i) + ".csv")
                Data = pd.read_csv(FileName, header = 0, index_col = 0)

                self.DataSet[plus + 9 + i, 0, :] = np.array(Data.ix[0,:])
                self.DataSet[plus + 9 + i, 1:, :] = np.array(Data.ix[:,:])
        
            plus += 32
        
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
        self.Input_Gen = 100                                                    # Shape of the Generative network input
        self.Input_Dis = (self.DataSet.shape[1:])                               # Shape of the Discriminative network input = [Rows, Cols, Chan].
            # Hidden layers outputs.
        self.Depth_Dis = 64                                                     # Initial depth of the Discriminative layers.                      
        self.Depth = 256                                                        # Initial depth of the Generative layers.
        self.Width = 5                                                          # Initial width of the Generative network.
        self.Length = 2                                                         # Initial length of the Generative network.
            # Convolutionals layers parameters.
        self.Kernel = 4
        self.strides = 2
            # Neural networks train parameters.
        self.Dropout_rate = 0.1                                                 # Percentage to dropout the input data.
        self.Momentum = 0.2
    
    def Plot_Loss(self, Losses):
        '''
        Function that plot the Adversarial and Discriminator models Losses.
        '''
        plt.figure(figsize=(10,8))
        plt.plot(Losses['D'], label = 'Discriminative loss')
        plt.plot(Losses['A'], label = 'Adversarial loss')
        plt.legend()
        plt.show()

    def Plot_Acc(self, Acc):
        '''
        Function that plot the Adversarial and Discriminator models accuracies.
        '''
        plt.figure(figsize=(10,8))
        plt.plot(Acc['D'], label = 'Discriminative accuracy')
        plt.plot(Acc['A'], label = 'Adversarial accuracy')
        plt.legend()
        plt.show()     
        
    def CreateGen(self):
        '''
        Function that create the model of the Generative network.
        '''
        # Is created the Generative network.
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
         
        # Output layer.
        Gen.add(Conv2DTranspose(1, self.Kernel, border_mode = 'same'))
        Gen.add(Activation('tanh'))
        
        # Model summary.
        print('Generative network model:'+'\n'*2); Gen.summary(line_length = 100, positions = [.45, .7, 2, 1.]); print('\n'*2)
        
        return Gen

    def CreateDis(self):
        '''
        Function that create the model of the Discriminative network.
        '''
        # Is created the Discriminative network.
        Dis = Sequential()
        
        # First layer of the network.
        Dis.add(Conv2D(self.Depth_Dis, self.Kernel, strides = int(self.strides/self.strides), input_shape = self.Input_Dis, padding = 'same'))
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
        Dis.add(Conv2D(self.Depth_Dis*8, self.Kernel, strides = self.strides, padding = 'same'))
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
        
        # Model summary.
        print('Discriminative network model:'+'\n'*2); Dis.summary(); print('\n'*2)
        
        return Dis
    
    def DiscriminativeModel(self, LearningRate_D, Decrement_D):
        '''
        Function that implement the Discriminator model.
        '''
        # Model used to saved the Discriminative network.
        D = Sequential()
        
        # Is added the Discriminative network.
        D.add(self.CreateDis())

        # Is definite the network training.
        RMS_D = RMSprop(lr = LearningRate_D, decay = Decrement_D)
        D.compile(loss = 'binary_crossentropy', optimizer = RMS_D, metrics = ['accuracy'])
        
        return D
    
    def AdversarialModel(self, Generator, Discriminator, LearningRate_A, Decrement_A):
        '''
        Function that implement the Adversarial model (Generative + Discriminative)
        '''
        # Is created a model to saved the both netwroks.
        A = Sequential()
        
        # Is added the Generative network. 
        A.add(Generator)
        
        # Is added the Discriminative network.
        A.add(Discriminator)
        
        # Is definet the network training.
        RMS_A = RMSprop(lr = LearningRate_A, decay = Decrement_A)
        A.compile(loss = 'binary_crossentropy', optimizer = RMS_A, metrics = ['accuracy'])
        
        # Model summary.
        print('Adversarial network model:'+'\n'*2); A.summary(); print('\n'*2)
        
        return A
    
    def Train(self, Iterations = 10, Batch_Size = 32, Version = str('1'), Plot_Freq = 20, LearningRate_A = 0.0001, Decrement_A = 3e-8, LearningRate_D = 0.0002, Decrement_D = 6e-8):
        '''
        Function that train the GAN model.
        First is trained the Discriminator model, using the original motions 
        sequences and motion sequences created by the Generative network without 
        be trained; then is trained the Adversarial model (Gen + Dis) to 
        increase the performer of the Generative network.
        '''
        X_Train = self.DataSet
        
        # Are created the models of the neural network.
        Generator = self.CreateGen()
        Discriminator = self.DiscriminativeModel(LearningRate_D, Decrement_D)
        Adversarial = self.AdversarialModel(Generator, Discriminator, LearningRate_A, Decrement_A)
        
        ListAnimations = np.empty([Iterations, Batch_Size, 39, 16])             # List used to save the new motion sequences.
        
        Losses = {'D':[], 'A':[]}                                               # List used to save the discriminator and the adversarial models losses.
        Accs = {'D':[], 'A':[]}                                                 # List used to save the discriminator and the adversarial models accuracies.
        
        # The networks are trained.
        for I in range(Iterations):
            Index = np.random.randint(X_Train.shape[0], size = Batch_Size)      # List used to saved the index of the motion sequence selected.
             
            # Are loaded the animations in a random order.
            Animations = X_Train[Index, :, :, :]
 
            # Is created the input of the Generative model.
            Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
             
            # Are created new motion sequences.
            FakeAnimations = Generator.predict(Noise)
             
            # Is created the training set to the Discriminator model.
            X = np.concatenate((Animations, FakeAnimations))
            Y = np.ones([2*Batch_Size, 1]); Y[Batch_Size:, :] = 0
             
            # The Discriminator model is trained 
            Dis_loss = Discriminator.train_on_batch(X, Y)
             
            # Is created the training set to the Adversarial model.
            Adv_Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
            Adv_Y = np.ones([Batch_Size, 1])
             
            # The Adversarial model is trained.
            Adv_loss = Adversarial.train_on_batch(Adv_Noise, Adv_Y)
             
            # Is showed the evolution of the training.
            log_mesg = "%d: [D loss: %f, acc: %f]" % (I, Dis_loss[0], Dis_loss[1])
            log_mesg = "%s: [A loss: %f, acc: %f]" % (log_mesg, Adv_loss[0], Adv_loss[1])
            print(log_mesg)
             
            # Is saved the discriminator and the adversarial models losses to be plotted.
            Losses['D'].append(Dis_loss[0])
            Losses['A'].append(Adv_loss[0])
             
            # Is saved the discriminator and the adversarial models accuracies to be plotted.
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
        Adversarial.save('D:\Tesis\Python\Liclipse\Tesis\ImplementRNAs\Prueba_GAN\Adversarial Model')
        Discriminator.save('D:\Tesis\Python\Liclipse\Tesis\ImplementRNAs\Prueba_GAN\Discriminative Model')
        Generator.save('D:\Tesis\Python\Liclipse\Tesis\ImplementRNAs\Prueba_GAN\Generative Model')
  
        # Are saved the new motion sequences in .cvs files.
        for L in range(len(ListAnimations)):
            for A in range(len(FakeAnimations)) :
                DataFrame = pd.DataFrame(ListAnimations[L, A, :, :])
                DataFrame.to_csv("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorSecuenceOfMovements\DataBaseGeneratedByRNA\ NewAnimation " + Version + " " + str(L+1) + "-" + str(A+1) + ".csv", sep = ",", header = False, index = False, index_label = 'Node')
          
if __name__ == '__main__':
    
    GAN = DCGAN()
    Timer = TakeTime()
    GAN.Train(Iterations = 1000, Batch_Size = 32, Version = str('1'), Plot_Freq = 1000, LearningRate_A = 0.0002, Decrement_A = 3e-2, LearningRate_D = 0.0002, Decrement_D = 6e-8)
    Timer.Time()

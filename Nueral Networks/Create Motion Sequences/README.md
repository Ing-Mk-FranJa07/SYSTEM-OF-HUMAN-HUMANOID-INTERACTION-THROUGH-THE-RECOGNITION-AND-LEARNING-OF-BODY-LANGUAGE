# Generative Adversarial Network (GAN): Neural network model used to increase the motion sequences data base created to perform the interaction between the Robot Pepper and the humans.

### Description:

The [Generative Adversarial Network](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) is a recent development in deep learning introduced by Ian Goodfellow en 2014. This model attends the problem of unsupervised learning by training two deep networks called Generator and Discriminator. The both networks compete and cooperate with the other to learn how to perform their tasks.

The GAN model has been explained in the most of the times like a case of a counterfeiter (Generative network) and a policeman (Discriminator network), initially the counterfeiter creates and show to the policeman a "fake money" (the data); the policeman says that it's fake or real, and gives feedback to the counterfeiter why the money is fake. Now the counterfeiter tries to make new fake money based on the feedback that it received, and show to the policeman the fake money again. Policeman decide if the money is or not fake and offers a new feedback to counterfeiter. This cycle continues indefinitely while the counterfeiter goes to creating each time a better fake money and it will be looking so similar to the real money, finally the policeman is fooled.

Theoretically the GAN is very simple, but build a model that works correctly is very difficult, because there are two deep networks coupled together making back propagation of gradients twice as challenging. Exist an interesting example of the application to the GAN models using [Deep Convolutional GAN (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf%C3%AF%C2%BC%E2%80%B0) that demonstrated how to build a practical GAN to learn by itself how to synthesize new images.

In this case, the script [RNA_Generate_Animations_GAN.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Create%20Motion%20Sequences/RNA_Generate_Animations_GAN.py) build a GAN model used to try to create new motion sequences, and not images, very similar to the original motion sequences that was created to be played to the [Robot Pepper](https://www.ald.softbankrobotics.com/en/robots/pepper). Was decided try to build a GAN model, because the structure of the data is a matrix similar to the structure of a gray image.

The image shows a representation of the GAN model created.

![gan model](https://user-images.githubusercontent.com/31509775/35186134-2b8dab0e-fddd-11e7-8fad-0e60e305e5c2.PNG)

### Software requirements:

This script was developed using **PYHTON 3.6 (64 bits) in WINDOWS 10** and the following wrappers.

Is recommended install [Anaconda (Python 3.6 (64 bits) version](https://www.anaconda.com/download/#windows) to get easier the packages necessaries. 

* [keras version 2.0.6](https://keras.io/#installation): keras website link with all installation instructions.
* [Tensorflow version 1.2.1 (keras backend engine)](https://www.tensorflow.org/install/install_windows): Tensorflow windows installation instructions link.
* [pandas version 0.19.2](https://pandas.pydata.org/): pandas website link, the download option to the version 0.19.2 is there.
* numpy version 1.11.3
* matplotlib 2.0.0

To use correctly this script, please consider the follow steps.

### First step: Create Data Base.

It's necessary to have a data base, and it's possible create it with the tool: [AnimationDataBaseCreator.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Motion%20Sequences%20Data%20Base%20Creator), this tool generate a .csv file in which is saved a matrix that containing 40 rows and 17 columns (The first row has the header and the first column has the name of the motion sequence), there are 16 columns that have the information about a specific joint of the Robot Pepper; and there are 39 angles values (rads) of each joint. The [Animation 1.csv](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Motion%20Sequences%20Data%20Base%20Creator/Motion%20Sequences/Animation%201.csv) is an example of a motion sequence developed by the author with the tool.

For the implementation of the system developed [Recognition_And_Learning_BodyLenguage_System.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project) were created 10 [motion sequences](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Motion%20Sequences%20Data%20Base%20Creator/Motion%20Sequences) used to the Robot Pepper interact with the humans being coherent with their mood, and 22 used to the Robot Pepper interactue with the humans being coherente with the conversation.

### Second step: Neural Network Structure.

The script developed has two classes, the first class "TakeTime" is used to calculate the time spent to train the GAN model. The second class "DCGAN" is used to build and train the GAN model, also save the models and the motion sequences created by the GAN model; the init function allow load the data and organize it in the structure used by the model.

**The data structure** used by the model is an array that contain the data in a matrix that has: (40 rows, 16 columns, 1 channel), like a gray image = 1 channel, to generate this, the whole data, that is to say, all motion sequences are grouped, and to each sequence is added one raw copying the first raw. This is done, because the structure of the Generative model goes increasing the kernel in a multiple of 2. Also the data is normalize, dividing the whole data by the maximun value, to put the all values in the interval -1.0 to 1.0.

```python
    96        # Set up the input data.
    97        self.DataSet = self.DataSet.reshape(self.DataSet.shape[0], self.DataSet.shape[1], self.DataSet.shape[2], 1)
    98        self.DataSet = self.DataSet.astype('float32')
    99
   100        # Is calculated the maximum value of the input data.
   101        self.DataMax = np.max(self.DataSet)
   102        
   103        # Normalize the input data; -1.0 <= input data <= 1.0.
   104        self.DataSet = self.DataSet / self.DataMax
```

**WARNINGS**

* Please make sure of the path that has the address of the .csv files be correct in the lines 80 and 88 (don't change or delete the files names that are written after the last slash):

```python
   76         # Is loaded the motion sequences.
   77         for D in range(Repetition):
   78             # Motion sequences used to represent emotions:
   79             for i in range(1,11):                
   80                 FileName = str("...\Motion_Sequences\ Emotion " + str(i) + ".csv")
   81                 Data = pd.read_csv(FileName, header = 0, index_col = 0)
   82
   83                 self.DataSet[plus + i - 1, 0, :] = np.array(Data.ix[0,:])
   84                 self.DataSet[plus + i - 1, 1:, :] = np.array(Data.ix[:,:])
   85
   86             # Motions sequences used to used in conversations.
   87             for i in range(1,23):                 
   88                 FileName = str("...\Motion_Sequences\ Animation " + str(i) + ".csv")
   89                 Data = pd.read_csv(FileName, header = 0, index_col = 0)
   90
   91                 self.DataSet[plus + 9 + i, 0, :] = np.array(Data.ix[0,:])
   92                 self.DataSet[plus + 9 + i, 1:, :] = np.array(Data.ix[:,:])
   93
   94            plus += 31
```

* Take care with the parameters to handling the long of the data set in the lines 72-74.

```python
   71       # Parameters used to build the input data set.
   72        Repetition = 1                                                          # Variable used to increase the original animation in the input data.
   73        plus = 0                                                                # Variable used to increase the original animation in the input data.
   74        self.DataSet = np.empty([Repetition*32,40,16])                          # Input Data.
```

The class "DCGAN" has a function named "CreateGen" which build the **Generative network model**, this network synthesizes the "fake" motion sequences. The fake motion sequence is generated from a 100-dimensional noise, that has a uniform distribution between -1.0 to 1.0 using the inverse of convolution, transposed convolution. Between the layers, batch normalization is used to stabilizes learning, is used the upsampling between the first three layers because it synthesizes better the data. The activation function after each layer is the Hiperbolic tangent "tanh", because its output take a real value between -1.0 to 1.0 (same interval that the originals motion sequences); the droput is used in the first layer to prevents over fitting. 

```python
   145    def CreateGen(self):
   146        '''
   147        Function that create the model of the Generative network.
   148        '''
   149        # Is created the Generative network.
   150        Gen = Sequential()
   151        
   152        # First layer of the network.
   153        Gen.add(Dense(self.Width* self.Length* self.Depth, input_dim = self.Input_Gen))
   154        Gen.add(BatchNormalization(momentum = self.Momentum))
   155        Gen.add(Activation('tanh'))
   156        Gen.add(Reshape((self.Width, self.Length, self.Depth))) 
   157        Gen.add(Dropout(self.Dropout_rate))
   158        
   159        # Second layer of the network.
   160        Gen.add(UpSampling2D(size = (2, 2)))
   161        Gen.add(Conv2DTranspose(int(self.Depth/2), self.Kernel, border_mode = 'same'))
   162        Gen.add(BatchNormalization(momentum = self.Momentum))
   163        Gen.add(Activation('tanh'))
   164        
   165        # Third layer of the network.
   166        Gen.add(UpSampling2D(size = (2, 2)))
   167        Gen.add(Conv2DTranspose(int(self.Depth/4), self.Kernel, border_mode = 'same'))
   169        Gen.add(BatchNormalization(momentum = self.Momentum))
   170        Gen.add(Activation('tanh'))
   171        
   172        # Fourth layer of the network.
   173        Gen.add(UpSampling2D(size = (2, 2)))
   174        Gen.add(Conv2DTranspose(int(self.Depth/8), self.Kernel, border_mode = 'same'))
   175        Gen.add(BatchNormalization(momentum = self.Momentum))
   176        Gen.add(Activation('tanh'))
   177       
   178        # Output layer.
   179        Gen.add(Conv2DTranspose(1, self.Kernel, border_mode = 'same'))
   180        Gen.add(Activation('tanh'))
   181
   182        # Model summary.
   183        print('Generative network model:'+'\n'*2); Gen.summary(line_length = 100, positions = [.45, .7, 2, 1.]); print('\n'*2)
   184      
   185        return Gen
```

* The image represents the structure of the Generative network.
![generative model7](https://user-images.githubusercontent.com/31509775/32303654-e7969c1e-bf37-11e7-83f8-d0871afc6ae4.PNG)

The function named "CreateDis" build the **Discriminative network model** which decide if the data is real (original motion sequences) or fake (motion sequences created by the generative network) and is a deep convolutional neural network. The input is a matrix that follows the structure (40 rows x 16 columns x 1 channel), the output of this model is obtained with the sigmoid function that determine the probability of how real is the data; (0.0 = complete fake, 1.0 = complete real); this model is different to a typical convolutional network is the absence of max-pooling bewteen layers, instead is used a stride convolution for down sampling. The activation function used in each convolutional layer is leaky Relu, and the dropout between layers prevent overfitting and memorization.

```python
   186    def CreateDis(self):
   187        '''
   188        Function that create the model of the Discriminative network.
   189        '''
   190        # Is created the Discriminative network.
   191        Dis = Sequential()
   192        
   193        # First layer of the network.
   194        Dis.add(Conv2D(self.Depth_Dis, self.Kernel, strides = int(self.strides/self.strides), input_shape = self.Input_Dis, padding = 'same'))
   195        Dis.add(LeakyReLU(alpha = 0.2))
   196        Dis.add(Dropout(self.Dropout_rate))
   197        
   198        # Second layer of the network.
   199        Dis.add(Conv2D(self.Depth_Dis*2, self.Kernel, strides = self.strides, padding = 'same'))   
   200        Dis.add(LeakyReLU(alpha = 0.2))
   201        Dis.add(Dropout(self.Dropout_rate))
   202        
   203        # Third layer of the network.
   204        Dis.add(Conv2D(self.Depth_Dis*4, self.Kernel, strides = self.strides, padding = 'same'))
   205        Dis.add(LeakyReLU(alpha = 0.2))
   206        Dis.add(Dropout(self.Dropout_rate))
   207         
   208        # Fourth layer of the network.
   209        Dis.add(Conv2D(self.Depth_Dis*8, self.Kernel, strides = self.strides, padding = 'same'))
   210        Dis.add(LeakyReLU(alpha = 0.2))
   211        Dis.add(Dropout(self.Dropout_rate))
   212
   213        # Fifth layer of the network.
   214        Dis.add(Conv2D(self.Depth_Dis*16, self.Kernel, strides = int(self.strides/self.strides), padding = 'same'))
   215        Dis.add(LeakyReLU(alpha = 0.2))
   216        Dis.add(Dropout(self.Dropout_rate))
   217
   218        # Output layer.
   219        Dis.add(Flatten())
   220        Dis.add(Dense(1))
   221        Dis.add(Activation('sigmoid'))
   222
   223        # Model summary.
   224        print('Discriminative network model:'+'\n'*2); Dis.summary(); print('\n'*2)
   225
   226        return Dis
```

* The image represent the structure of the Discriminative network.
![discriminative model](https://user-images.githubusercontent.com/31509775/32345300-c775c43c-bfd7-11e7-824c-e1d53ca4f967.PNG)

Before the start with the training of the neural networks, is necessary create the GAN model, and for this is necessary two models the first one is the **Discriminator model** (DiscriminatorModel function) that is the discriminative network with the loss function definied, the second model is the **Adversarial model** (AdversarialModel function) that is the generative and the discriminative networks stacked together; the generative part is trying to cheat the discriminative and learning from its feedback at the same time. The models use the binary cross entropy like optimization function, and RMSprop optimization algorithm.

**WARNINGS**

Finally, the GAN model is trained. **The GAN model training** is developed following two steps in each epoch. First, is necessary train the discriminator, showing it some examples of the real data and some examples of the fake data created by the generative network using just noise; the second step is train the generative network via the chained models, that is to say, train the adversarial model generating sample data and try to push the chained generative network and the discriminative network to tell if the data is real or not; however is necessary don't alter the weights of the discriminative network during this step, so that's why the training of the discriminative network is freeze. 

```python
   286        # The networks are trained.
   287        for I in range(Iterations):
   288            Index = np.random.randint(X_Train.shape[0], size = Batch_Size)      # List used to saved the index of the motion sequence selected.
            
   289            # Are loaded the animations in a random order.
   290            Animations = X_Train[Index, :, :, :]
   291
   292            # Is created the input of the Generator model.
   293            Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
   294          
   295            # Are created new motion sequences.
   296            FakeAnimations = Generator.predict(Noise)
   297           
   298            # Is created the training set to the Discriminator model.
   299            X = np.concatenate((Animations, FakeAnimations))
   300            Y = np.ones([2*Batch_Size, 1]); Y[Batch_Size:, :] = 0
   301           
   302            # The Discriminator model is trained 
   303            Dis_loss = Discriminator.train_on_batch(X, Y)
   304           
   305            # Is created the training set to the Adversarial model.
   306            Adv_Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
   307            Adv_Y = np.ones([Batch_Size, 1])
   308          
   309            # The Adversarial model is trained.
   310            Adv_loss = Adversarial.train_on_batch(Adv_Noise, Adv_Y)
   311        
   312            # Is showed the evolution of the training.
   313            log_mesg = "%d: [D loss: %f, acc: %f]" % (I, Dis_loss[0], Dis_loss[1])
   314            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, Adv_loss[0], Adv_loss[1])
   315            print(log_mesg)
```

* The image represents the train loop of the GAN model.
![gan training loop](https://user-images.githubusercontent.com/31509775/32347817-e3cae3b2-bfdf-11e7-9786-eae586f0dbf8.PNG)

* The next image shows the loss and accuracy of the Adversarial and Generator models built and trained by the author after 1000 epochs.
![acc and loss gan](https://user-images.githubusercontent.com/31509775/35189625-da20310c-fe1c-11e7-9d84-c92d473011c7.PNG)

After the GAN model has been trained, the models of the networks: Generative, Discriminative and Adversarial, are saved. Also, the motion sequences created in the training process are saved to, after they have been reshaped and denormalize to have the structure of the original data. 

```python
   326           # Is denormalize the new motion sequences.
   327            FakeAnimations = FakeAnimations.reshape(FakeAnimations.shape[0], FakeAnimations.shape[1], FakeAnimations.shape[2])
   328            FakeAnimations = FakeAnimations[:, 1: ,:]
   329            FakeAnimations = FakeAnimations * self.DataMax
```

**WARNING**

* Please make sure of the path that has the address of the models and the .csv files be correct in the follow lines (don't change or delete the files names that are written after the last slash):

```python
   339        # Are saved the nueral networks models.
   340        Adversarial.save('...\Adversarial Model')
   341        Discriminator.save('...\Discriminator Model')
   342        Generator.save('...\Generator Model')
   343
   344        # Are saved the new motion sequences in .cvs files.
   345        for L in range(len(ListAnimations)):
   346           for A in range(len(FakeAnimations)) :
   347                DataFrame = pd.DataFrame(ListAnimations[L, A, :, :])
   348                DataFrame.to_csv("...\DataBaseGeneratedByRNA\ NewAnimation " + Version + " " + str(L+1) + "-" + str(A+1) + ".csv", sep = ",", header = False, index = False, index_label = 'Node')
```

The GAN model presented here, can develop ["originals" motion sequences](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Create%20Motion%20Sequences/DataBaseGeneratedByRNA), but this sequences are not exactly to the motion sequences developed by the author, whatever, tuning more the model, it's possible to get a better result.

```python
   350 if __name__ == '__main__':
   351     
   352     GAN = DCGAN()
   353     Timer = TakeTime()
   354     GAN.Train(Iterations = 1000, Batch_Size = 32, Version = str('1'), Plot_Freq = 1000, LearningRate_A = 0.0002, Decrement_A = 3e-2, LearningRate_D = 0.0002, Decrement_D = 6e-8)
   355     Timer.Time()
```

* The image shows a comparation between a motion sequences created by the GAN model designed in the first epoch and in the last epoch of the training. It can be seen the change of the erratic values generated to a soft sequence of angles values.

![fist and last epoch motion sequence](https://user-images.githubusercontent.com/31509775/35189633-264665e2-fe1d-11e7-9512-0d091d263d1a.png)

* The image shows an animation motion sequence created by the user and the motion sequence created by the GAN model.

![comparing motion sequences](https://user-images.githubusercontent.com/31509775/35190041-508c57b8-fe26-11e7-9848-85b4b8aa1ee3.gif)

**Click on the images to see them with better quality**

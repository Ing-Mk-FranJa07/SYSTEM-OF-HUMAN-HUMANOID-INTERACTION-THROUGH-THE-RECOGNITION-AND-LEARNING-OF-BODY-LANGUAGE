# Generative Adversarial Network (GAN): Neural network model used to increase the motion sequences data base created to perform the interaction between the Robot Pepper and the humans.

### Description:

The [Genrative Adversarial Network](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) is a recent development in deep learning introduced by Ian Goodfellow en 2014. This model attend the problem of unsupervised learning by training two deep networks called Generator and Discriminator. The both networks compete and cooperate with the other to learn how to perform their tasks.

The GAN model has been explained in the most of the times like a case of a counterfeiter (Generative network) and a policeman (Discriminator network), initially the counterfeiter create and show to the policeman a "fake money" (the data); the policeman says that it's fake or real, and gives feedback to the counterfeiter why the money is fake. Now the counterfeiter trys to make new fake money based on the feedback that it received, and show to the policeman the fake money again. Policeman decide if the money is or not fake and offers a new feedback to counterfeiter. This cycle continues indefinitely while the counterfeiter goes to creating each time a better fake money and it will be looking so similar to the real money, finally the policeman is fooled.

Teoretically the GAN is very simple, but build a model that works correctly is very difficult, because there are two deep networks coupled together making back propagation of gradients twice as challenging. Exist an interesting example of the application to the GAN models using [Deep Convolutional GAN (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf%C3%AF%C2%BC%E2%80%B0) that demonstrated how to build a practical GAN to learn by itselvef how to synthesize new images.

In this case, the script [RNA_Generate_Animations_GAN.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Create%20Motion%20Sequences/RNA_Generate_Animations_GAN.py) build a GAN model used to try to create new motion sequences, and not images, very similar to the original motion sequences that was created to be played to the [Robot Pepper](https://www.ald.softbankrobotics.com/en/robots/pepper). Was decided try to build a GAN model, because the structure of the data is a matrix similar to the structure of the a gray image.

The image show a representation of the GAN model created.

![gan model](https://user-images.githubusercontent.com/31509775/32303734-6251d8c4-bf38-11e7-8555-b827076c2108.PNG)

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

For the implementation of the system developed [Recognition_And_Learning_BodyLenguage_System.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project) were created 10 [motion sequences](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Motion%20Sequences%20Data%20Base%20Creator/Motion%20Sequences) used to the Robot Pepper interactue with the humans being coherent with their mood, and 22 used to the Robot Pepper interactue with the humans being coherente with the conversation.

### Second step: Neural Network Structure.

The script developed has two classes, the first class "TakeTime" is used to calculate the time spent to train the GAN model. The second class "DCGAN" is used to build and train the GAN model, also save the models and the motion sequences created by the GAN model; the init function allow load the data and organize it in the structure used by the model.

**The data structure** used by the model is an array that contain the data in a matrix that has: (40 rows, 16 columns, 1 channel), like a gray image = 1 channel, to generate this, the whole data, that is to say, all movement sequences are grouped, and to each sequences is added one raw copying the first raw. This is done, because the structure of the Generative model goes increasing the kernel in a multiple of 2. Also the data is normalize, dividing the whole data by the maximmun value, to put the all values in the interval -1.0 to 1.0.

```python
   114        # Set up the input data.
   115        self.DataSet = self.DataSet.reshape(self.DataSet.shape[0], self.DataSet.shape[1], self.DataSet.shape[2], 1)
   116        self.DataSet = self.DataSet.astype('float32')
   117
   118        # Is calculated the maximum value of the input data.
   119        self.DataMax = np.max(self.DataSet)
   120        
   121        # Normalize the input data; -1.0 <= input data <= 1.0.
   122        self.DataSet = self.DataSet / self.DataMax
```

**WARNINGS**

* Please make sure of the path that has the address of the .csv files be correct in the lines 80 and 97 (don't change or delete the files names that are written after the last slash):

* The code has a few commented lines (87-95, 106-112), that you can use or change to add copies of the 32 motion sequences created, this is because the author was traying to generete better motion sequences and the data set is pretty short, but the advice is try to create more and more motion sequences, or generete a random noise to the motion sequences created trying to have more data; also, you can load the same data several times to increase the data set.

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
   86     #             self.DataSet[i - 1, 0, :] = np.array(Data.ix[0,:])
   87     #             self.DataSet[i - 1, 1:, :] = np.array(Data.ix[:,:])
   88
   89 #                 for D in range(Repetition):#                      
   90 #                     self.DataSet[plus + D, 0, :] = np.array(Data.ix[0,:])
   91 #                     self.DataSet[plus + D, 1:, :] = np.array(Data.ix[:,:])
   92 #              
   93 #                 plus += Repetition
   94
   95             # Motions sequences used to used in conversations.
   96             for i in range(1,23):                 
   97                 FileName = str("...\Motion_Sequences\ Animation " + str(i) + ".csv")
   98                 Data = pd.read_csv(FileName, header = 0, index_col = 0)
   99
  100                 self.DataSet[plus + 9 + i, 0, :] = np.array(Data.ix[0,:])
  101                 self.DataSet[plus + 9 + i, 1:, :] = np.array(Data.ix[:,:])
  102
  103     #             self.DataSet[9 + i, 0, :] = np.array(Data.ix[0,:])
  104     #             self.DataSet[9 + i, 1:, :] = np.array(Data.ix[:,:])
  105
  106 #                 for D in range(Repetition):#                   
  107 #                     self.DataSet[plus + D, 0, :] = np.array(Data.ix[0,:])
  108 #                     self.DataSet[plus + D, 1:, :] = np.array(Data.ix[:,:])
  109 #                  
  110 #                 plus += Repetition
  111
  112 #            plus += 31
```

* Take care with the parameters to handling the long of the data set in the lines 72-74.

```python
   71       # Parameters used to build the input data set.
   72        Repetition = 1                                                          # Variable used to increase the original animation in the input data.
   73        plus = 0                                                                # Variable used to increase the original animation in the input data.
   74        self.DataSet = np.empty([Repetition*32,40,16])                          # Input Data.
```

The class "DCGAN" has a function named "CreateGen" which build the **Generative network model**, this network synthesizes the "fake" motion sequences. The fake motion sequence is generated from a 100-dimensional noise, that has a uniform distribution between -1.0 to 1.0) using the inverse of convolution, transposed convolution. In between the layers, batch normalization is used to stabilizes learning, is used the upsampling between the first three layers because it synthesizes better the data. The activation function after each layer is the Hiperbolic tangent "tanh", because its output take a real value between -1.0 to 1.0 (same interval that the originals motion sequences); the droput is used in the first layer to prevents over fitting. 

```python
   191    def CreateGen(self):
   192        '''
   193        Function that create the model of the Generative network.
   194        '''
   195        # Is created the Generative network.
   196        Gen = Sequential()
   197        
   198        # First layer of the network.
   199        Gen.add(Dense(self.Width* self.Length* self.Depth, input_dim = self.Input_Gen))
   200        Gen.add(BatchNormalization(momentum = self.Momentum))
   201        Gen.add(Activation('tanh'))
   202        Gen.add(Reshape((self.Width, self.Length, self.Depth))) 
   203        Gen.add(Dropout(self.Dropout_rate))
   204        
   205        # Second layer of the network.
   206        Gen.add(UpSampling2D(size = (2, 2)))
   207        Gen.add(Conv2DTranspose(int(self.Depth/2), self.Kernel, border_mode = 'same'))
   208        Gen.add(BatchNormalization(momentum = self.Momentum))
   209        Gen.add(Activation('tanh'))
   210        
   211        # Third layer of the network.
   212        Gen.add(UpSampling2D(size = (2, 2)))
   213        Gen.add(Conv2DTranspose(int(self.Depth/4), self.Kernel, border_mode = 'same'))
   214        Gen.add(BatchNormalization(momentum = self.Momentum))
   215        Gen.add(Activation('tanh'))
   216        
   217        # Fourth layer of the network.
   218        Gen.add(UpSampling2D(size = (2, 2)))
   219        Gen.add(Conv2DTranspose(int(self.Depth/8), self.Kernel, border_mode = 'same'))
   220        Gen.add(BatchNormalization(momentum = self.Momentum))
   221        Gen.add(Activation('tanh'))
   222       
   223        # Output layer.
   224        Gen.add(Conv2DTranspose(1, self.Kernel, border_mode = 'same'))
   235        Gen.add(Activation('tanh'))
   226        
   227      return Gen
```

* The image represent the structure of the Generative network.
![generative model7](https://user-images.githubusercontent.com/31509775/32303654-e7969c1e-bf37-11e7-83f8-d0871afc6ae4.PNG)

The function named "CreateDis" build the **Discriminative network model** wchic decide if the data is real (original motion sequences) or fake (motion sequences created by the generative network) and is a deep convolutional neural netwrok. The input is a matrix that has the follow structure (40 rows x 16 columns x 1 channel), the output of this model is optained with the sigmoid function that determine the probability of how real is the data; (0.0 = complety fake, 1.0 = complety real); this model is different to a typical convolutional network is the absence of max-pooling in bweteen layers, instead is used a strided convolution for downsampling. The activation function used in each convolutional layer is leaky Relu, and the dropout between layers prevent over fitting and memorization.

```python
   229    def CreateDis(self):
   230        '''
   231        Function that create the model of the Discriminative network.
   232        '''
   233        # Is created the Discriminative network.
   234        Dis = Sequential()
   235        
   236        # First layer of the network.
   237        Dis.add(Conv2D(self.Depth_Dis, self.Kernel, strides = self.strides, input_shape = self.Input_Dis, padding = 'same'))
   238        Dis.add(LeakyReLU(alpha = 0.2))
   239        Dis.add(Dropout(self.Dropout_rate))
   240        
   241        # Second layer of the network.
   242        Dis.add(Conv2D(self.Depth_Dis*2, self.Kernel, strides = self.strides, padding = 'same'))   
   243        Dis.add(LeakyReLU(alpha = 0.2))
   244        Dis.add(Dropout(self.Dropout_rate))
   245        
   246        # Third layer of the network.
   247        Dis.add(Conv2D(self.Depth_Dis*4, self.Kernel, strides = self.strides, padding = 'same'))
   248        Dis.add(LeakyReLU(alpha = 0.2))
   249        Dis.add(Dropout(self.Dropout_rate))
   250         
   251        # Fourth layer of the network.
   252        Dis.add(Conv2D(self.Depth_Dis*8, self.Kernel, strides = self.strides, padding = 'same'))
   253        Dis.add(LeakyReLU(alpha = 0.2))
   254        Dis.add(Dropout(self.Dropout_rate))
   255
   256        # Fifth layer of the network.
   257        Dis.add(Conv2D(self.Depth_Dis*16, self.Kernel, strides = int(self.strides/self.strides), padding = 'same'))
   258        Dis.add(LeakyReLU(alpha = 0.2))
   259        Dis.add(Dropout(self.Dropout_rate))
   260
   261        # Output layer.
   262        Dis.add(Flatten())
   263        Dis.add(Dense(1))
   264        Dis.add(Activation('sigmoid'))
   265    
   266        return Dis
```

* The image represent the structure of the Discriminative network.
![discriminative model](https://user-images.githubusercontent.com/31509775/32345300-c775c43c-bfd7-11e7-824c-e1d53ca4f967.PNG)

Before the start with the training of the neural networks, is necessary create the GAN model, and for this is necesary two models the first one is the **Disciminator model** (DiscriminatorModel function) that is the discriminative network with the loss function definied, the second model is the **Adversarial model** (AdversarialModel function) that is the generative and the discriminative networks stacked together; the generative part is trying to foll the discriminative and learning from its feedback at the same time. The models uses the binary cross entropy like optimization function.

**WARNINGS**

* The optimizer algorithm can be choosen between the Adam, RMSProp and SGD for the both models changin the value (0, 1, 2) of the flag "OptimizerType" in the **line 399** where is called the main loop of the script. 

* The model's parameters can be configurated in the init function in the **lines 127-156**.

Finaly, the GAN model is trained. **The GAN model training** is developed following two steps in each epoch. First, is necessary train the discriminator, showing it some examples of the real data and some examples of the fake data created by the generative network using just noise; the second step is train the generative network via the chained models, that is to say, train the adversarial model generating sample data and try to push the chained generative network and the discriminative network to tell if the data is real or not; however is necessary don't alter the weights of the discriminative network during this step, so that's why the training of the discriminative network is freeze. 

```python
   332        # The networks are trained.
   333        for I in range(Iterations):
   334            Index = np.random.randint(X_Train.shape[0], size = Batch_Size)      # List used to saved the index of the motion sequence selected.
            
   335            # Are loaded the animations in a random order.
   336            Animations = X_Train[Index, :, :, :]
   337
   338            # Is created the input of the Generator model.
   339            Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
   340          
   341            # Are created new motion sequences.
   342            FakeAnimations = Generator.predict(Noise)
   343           
   344            # Is created the training set to the Discriminator model.
   345            X = np.concatenate((Animations, FakeAnimations))
   346            Y = np.ones([2*Batch_Size, 1]); Y[Batch_Size:, :] = 0
   347           
   348            # The Discriminator model is trained 
   349            Dis_loss = Discriminator.train_on_batch(X, Y)
   350           
   351            # Is created the training set to the Adversarial model.
   352            Adv_Noise = np.random.uniform(-1.0, 1.0, size = [Batch_Size, self.Input_Gen])
   353            Adv_Y = np.ones([Batch_Size, 1])
   354          
   355            # The Adversarial model is trained.
   356            Adv_loss = Adversarial.train_on_batch(Adv_Noise, Adv_Y)
   357        
   358            # Is showed the evolution of the training.
   359            log_mesg = "%d: [D loss: %f, acc: %f]" % (I, Dis_loss[0], Dis_loss[1])
   360            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, Adv_loss[0], Adv_loss[1])
   361            print(log_mesg)
```

* The image represente the train loop of the GAN model.
![gan training loop](https://user-images.githubusercontent.com/31509775/32347817-e3cae3b2-bfdf-11e7-9786-eae586f0dbf8.PNG)

* The next image show the loss and accuracy of the Adversarial and Generator models built and trained by the author after 300 epochs.
![acc and loss gan](https://user-images.githubusercontent.com/31509775/32348907-614a960e-bfe3-11e7-9fc7-21a152a0fba0.PNG)

After the GAN model has been trained, the models of the networks: Generative, Discriminative and Adversarial, are saved. Also, the motion sequences created in the training process are saved to, after they have been reshaped and denormalize to have the structure of the original data. 

```python
   371           # Is denormalize the new motion sequences.
   372            FakeAnimations = FakeAnimations.reshape(FakeAnimations.shape[0], FakeAnimations.shape[1], FakeAnimations.shape[2])
   373            FakeAnimations = FakeAnimations[:, 1: ,:]
   374            FakeAnimations = FakeAnimations * self.DataMax
```

**WARNING**

* Please make sure of the path that has the address of the models and the .csv files be correct in the follow lines (don't change or delete the files names that are written after the last slash):

```python
   385        # Are saved the nueral networks models.
   386        Adversarial.save('...\Adversarial Model ' + Version)
   387        Discriminator.save('...\Discriminator Model ' + Version)
   388        Generator.save('...\Generator Model ' + Version)
   389
   390        # Are saved the new motion sequences in .cvs files.
   391        for L in range(len(ListAnimations)):
   392           for A in range(len(FakeAnimations)) :
   393                DataFrame = pd.DataFrame(ListAnimations[L, A, :, :])
   394                DataFrame.to_csv("...\DataBaseGeneratedByRNA\ NewAnimation " + str(L+1) + "-" + str(A+1) + ".csv", sep = ",", header = False, index = False, index_label = 'Node')
```

The GAN model presented here, can develop ["originals" motion sequences](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Create%20Motion%20Sequences/DataBaseGeneratedByRNA), but this sequences are not similars to the motion sequences, you can change all the parameters used to buid and train the networks and also the optimizer algorithms, and try to get a better performance of this model.

```python
   395 if __name__ == '__main__':
   396     
   397     GAN = DCGAN()
   398     Timer = TakeTime()
   399     GAN.Train(Iterations = 300, Batch_Size = 32, Version = str('1'), shuffle = False, OptimizerType = 2, Plot_Freq = 300)
   400     Timer.Time()
```

* The image shows an animation motion sequence created by the user and the motion sequence created by the GAN model.

![pepper original motion sequence](https://user-images.githubusercontent.com/31509775/33086815-cb8823ba-ceb6-11e7-8dcf-6df2f62d1f71.gif)

**Click on the images to see them with better quality**

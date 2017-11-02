# Generative Adversarial Network (GAN): Neural network model used to increase the motion sequences data base created to perform the interaction between the Robot Pepper and the humans.

### Description:

The [Genrative Adversarial Network](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) is a recent development in deep learning introduced by Ian Goodfellow en 2014. This model attend the problem of unsupervised learning by training two deep networks called Generator and Discriminator. The both networks compete and cooperate with the other to learn how to perform their tasks.

The GAN model has been explained in the most of the times like a case of a counterfeiter (Generative network) and a policeman (Discriminator network), initially the counterfeiter create and show to the policeman a "fake money" (the data); the policeman says that it's face and gives feedback to the counterfeiter why the money is fake. Now the counterfeiter trys to make new fake money based on the feedback that it received, and show to the policeman the fake money again. Policeman decide if the money is or not fake and offers a new feedback to counterfeiter. This cycle continues indefinitely while the counterfeiter goes tcreating each time a better fake money and it will be looking so similar to the real money, finally the policeman is fooled.

Teoretically the GAN is very simple, but build a model that works correctly is very difficult, because there are two deep networks coupled together making back propagation of gradients twice as challenging. Exist an interesting example of the application to the GAN models using [Deep Convolutional GAN (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf%C3%AF%C2%BC%E2%80%B0) that demonstrated how to build a practical GAN to learn by itselvef how to synthesize new images.

In this case, the script [RNA_Generate_Animations_GAN.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Create%20Motion%20Sequences/RNA_Generate_Animations_GAN.py) build a GAN model used to try to create new motion sequences, and not images, very similar to the original motion sequences that was created to be played to the [Robot Pepper](https://www.ald.softbankrobotics.com/en/robots/pepper). Was decided try to build a GAN model, because the structure of the data is a matrix similar to the structure of the a gray image.

The image show a representation of the GAN model created.

![gan model](https://user-images.githubusercontent.com/31509775/32291493-3500cbd2-bf0b-11e7-8102-68e7da40dbc6.PNG)

This script was developed using **PYHTON 3.6 (64 bits) in WINDOWS 10** and the following wrappers.
* Keras version 2.0.6
* Tensorflow version 1.2.1 (keras backend engine)
* pandas version 0.19.2
* numpy version 1.11.3
* matplotlib 2.0.0

To use correctly this script, please consider the follow steps.

### First step: Create Data Base.

It's necessary to have a data base, and it's possible create it with the tool: [AnimationDataBaseCreator.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Motion%20Sequences%20Data%20Base%20Creator), this tool generate a .csv file in which is saved a matrix that containing 40 rows and 17 columns (The first row has the header and the first column has the name of the motion sequence), there are 16 columns that have the information about a specific joint of the Robot Pepper; and there are 39 angles values (rads) of each joint. The [Animation 1.csv](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Motion%20Sequences%20Data%20Base%20Creator/Motion%20Sequences/Animation%201.csv) is an example of a motion sequence developed by the author with the tool.

For the implementation of the system developed [Recognition_And_Learning_BodyLenguage_System.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project) were created 10 motion sequences used to the Robot Pepper interactue with the humans being coherent with their mood, and 22 motion sequences used to the Robot Pepper interactue with the humans being coherente with the conversation.

### Second step: Neural Network Structure.

The script [Generate_Animations_GAN.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Create%20Motion%20Sequences/RNA_Generate_Animations_GAN.py) has two classes, the first class "TakeTime" is used to calculate the time spent to train the GAN model. The second class "DCGAN" is used to build and train the GAN model, also save the models and the motion sequences created by the GAN model; the init function allow load the data and organize it in the structure used by the model.

The data structure used by the model is an array that contain the data in a matrix that has: (40 rows, 16 columns, 1 channel), like a gray image = 1 channel, to generate this, the whole data, that is to say, all movement sequences are grouped, and to each sequences is added one raw copying the first raw. This is done, because the structure of the Generative model goes increasing the kernel in a multiple of 2. Also the data is normalize, dividing the whole data by the maximmun value, to put the all values in the interval -1.0 to 1.0.
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
The class "DCGAN" has a function named "CreateGen" which build the Generative network model, this network synthesizes the "fake" motion sequences. The fake motion sequence is generated from a 100-dimensional noise, that has a uniform distribution between -1.0 to 1.0) using the inverse of convolution, transposed convolution. In between the layers, batch normalization is used to stabilizes learning, is used the upsampling between the first three layers because it synthesizes better the data. The activation function after each layer is the Hiperbolic tangent "tanh", because its output take a real value between -1.0 to 1.0 (same interval that the originals motion sequences); the droput is used in the first layer to prevents overfitting. 
```python
   191    def CreateGen(self):
   192        '''
   193        Function that create the model of the Generator network.
   194        '''
   195        # Is created the Generator network model.
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
   217       # Fourth layer of the network.
   218        Gen.add(UpSampling2D(size = (2, 2)))
   219        Gen.add(Conv2DTranspose(int(self.Depth/8), self.Kernel, border_mode = 'same'))
   220        Gen.add(BatchNormalization(momentum = self.Momentum))
   221        Gen.add(Activation('tanh'))
   222       
   223       # Output layer.
   224       Gen.add(Conv2DTranspose(1, self.Kernel, border_mode = 'same'))
   235       Gen.add(Activation('tanh'))
   226        
   227      return Gen
```
The image represent the structure of the Generative network.

![generative model7](https://user-images.githubusercontent.com/31509775/32303654-e7969c1e-bf37-11e7-83f8-d0871afc6ae4.PNG)



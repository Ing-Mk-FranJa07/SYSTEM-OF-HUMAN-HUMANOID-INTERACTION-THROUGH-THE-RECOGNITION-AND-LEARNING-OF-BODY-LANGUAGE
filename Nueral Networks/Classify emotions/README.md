
# Artificial Neural Network which classify the emotions using data from body posture.

### Description:

The script [RNA_Emotions_BodyPosture_Keras_Tensorflow.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Classify%20emotions/RNA_Emotions_BodyPosture_Keras_Tensorflow.py) was developed to create, train and test an artificial neural network which classify the mood of one person using like input 
a vector that containing 23 angles, each angle descrive the orientation of a specific joint of the body. 

This script was developed using **PYTHON 3.6 (64 bits) in WINDOWS 10** and the following wrappers.

* keras version 2.0.6
* Tensorflow version 1.2.1 (keras backend engine)
* pandas version 0.19.2
* numpy version 1.11.3
* matplotlib version 2.0.0
* sklearn version 0.18.2
* csv version 1.0.0

To use correctly this script, please consider the follow steps.

### First step: Create Data Base.

It's necesarry to have a data base, and it's possible create it with the tool: [DataBaseCreatorHumanPosture.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator), this tool generate a .csv file in which is saved a matrix that containing 25 joints angles and the description of the emotion that is being represented: [Angry 1](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Data%20Base%20Creator/Emotions%20DataBase/Angry/Angry%201.csv) is a sample of the matrix created by the tool.

For the implementation of the system developed: [RecognitionOfEmotions.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Recognition) were created 40 .csv files of each category: [Happy, Sad, Angry, Surprised, Reflexive and Normal](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator/Emotions%20DataBase). 

For the "Normal" category were created 30 .csv files.

### Second step: Organize the Data Base.

The script [DataBase_Organize.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Classify%20emotions/DataBase_Organize.py) was wrote to organize the data base unifying alls .csv files and creating two new .csv files, the first of them has the 80% of the whole data used to train the neural network and the second file has the 20% of the data used to test the neural network.

Also, this script add the codification to each category, in the both files, used like output of the neural network.

**WARNINGS**
* It's recommended extract the [.csv files created by the author](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator/Emotions%20DataBase) from each folder and save them in just one folder to use correctly the script written.
* Please make sure of the path that has the address of the .csv files be correct in the follow lines (don't change or delete the files names that are written after the last slash):
```[PYTHON]
    35     Category_Train = open("...\DataSet_Organized\Emotions_Unified\Postures_Train " + str(File_Code) + ".csv", "w") 
   ...
    38     Category_Test = open("...\DataSet_Organized\Emotions_Unified\Postures_Test " + str(File_Code) + ".csv", "w") 
   ...
    54        FileName = str("...\Emotions_Data_Base\ " + str(File_Category) + " " + str(F) + ".csv")            
   ...
    73    Postures_Train = open("...\DataSet_Organized\DataSetPostures_Train.csv", "w") 
   ...
    76    Postures_Test = open("...\DataSet_Organized\DataSetPostures_Test.csv", "w") 
   ...
    94        FileName = str("...\DataSet_Organized\Emotions_Unified\Postures_Train " + str(F) + ".csv")          
   ...
   101        FileName_Test = str("...\DataSet_Organized\Emotions_Unified\Postures_Test " + str(F) + ".csv")    
   ...
   116    Data = pd.read_csv("...\DataSet_Organized\DataSetPostures_" + str(File_Type) + ".csv", header = 0)
   ...
   143    Data.to_csv("...\DataSet_Organized\DataSetPostures__" + str(File_Type) + ".csv", header = True)
 ``` 
 * Note that in the lines 117-119 are being changed information into the .csv files; if you're working with the [.csv files created by the author](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator/Emotions%20DataBase), don't change or delet this lines. If you're working with new data created using the tool: [DataBaseCreatorHumanPosture.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator), these lines are not necessaries.
 ```[PYTHON]
     117    Data = Data_Train.replace("Opened", 1)
     118    Data = Data_Train.replace("Closed", 2)
     119    Data = Data_Train.replace("Unknow", 3)
 ```
### Third step: Neural Network structure.

The first part of the code in the script: [RNA_Emotions_BodyPosture_Keras_Tensorflow.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Classify%20emotions/RNA_Emotions_BodyPosture_Keras_Tensorflow.py) load the traininig and the testing data, creating the sets to train and test the neural network. 

**WARNING**
* Please make sure of the path that has the address of the .csv files be correct in the follow lines (don't change or delete the files names that are written after the last slash):
```[PYTHON]
   60 Data_Train = pd.read_csv("...\DataSet_Organized\DataSetPostures_Train.csv", header = 0, index_col = 0)
   61 Data_Test = pd.read_csv("...\DataSet_Organized\DataSetPostures_Test.csv", header = 0, index_col = 0)
```
The two first columns (angles: Head Roll and Head Pitch) are not used in the input training and testing set because they're not toasting information; also is created the ouput training and testing with the codification of each category.
```[PYTHON]
   64 X_Train = np.array(Data_Train.ix[:,2:25]) 
   65 Y_Train = np.array(Data_Train.ix[:,25:33])
   66 X_Test = np.array(Data_Test.ix[:,2:25])
   67 Y_Test = np.array(Data_Test.ix[:,25:33])
```
After that the Data has been loaded and schuffled in training and testing sets; is created the model of the neural network (lines 73-92) The neural network has 23 inputs and two hidden layers, the first of them has the same numbers of nodes that the inputs, and the second hidden layer has 17 nodes; the both layers have a Relu activation function. The ouput layer has 6 nodes and its activation function is sigmoidal because each node just can take two values: (0, 1) The total output is an "one hot" vector that contains six values; the neural network has like optimization function the categorical cross entropy loss, and it used the Stochastic downward gradient like the optimization type. 
```[PYTHON]
   73 # Are defined the parameters of the neural network and the model.
   74 HL_1_Nodes = 23
   75 HL_2_Nodes = 17
   76 RNA = Sequential()
   77
   78 # Is created the first hidden layer (Activation function = linear rectified, Nodes = 23) 
   79 RNA.add(Dense(HL_1_Nodes, activation = 'relu', input_shape = (X_Train.shape[1],))) 
   80 RNA.add(Dropout(0.05)) # Function that prevent the over training.
   81
   82 # Is created the second hidden layer (Activation function = linear rectified, Nodes = 17)
   83 RNA.add(Dense(HL_2_Nodes, activation = 'relu')) 
   84
   85 # Is created the output layer (Activation function = Sigmoidal, Nodes = 6)
   86 RNA.add(Dense(6, activation = 'sigmoid')) 
   87
   88 # Is defined the train parameters of the neural network.
   89 sgd = SGD(lr = 0.025, decay = 1e-5, momentum = 0.15, nesterov = False)
   90 RNA.compile(loss = 'categorical_crossentropy', # Optimization function.
   91             optimizer = sgd,                   # Optimization type: Stochastic downward gradient
   92             metrics = ['accuracy'])
```
The next image is a representation of the neural network designed.
![Ann classification problem](https://user-images.githubusercontent.com/31509775/32282186-e80f7e7a-beee-11e7-85a2-af58946356f5.PNG)

After that the neural network has been created, is trained and then is tested; the accuracy of the neural network designed by the author is the almost the 94%; to check the performance of the neural network; has been written a function that organize and show the cunfusion matrix; this function can show the matrix in two modes, porcentage or total data, just changing a boolean value ('Normalize') when it is called. 
```[PYTHON]
   29 def Plot_Confusion_Matrix(Matrix, Title, Cmap, Normalize = True):
   30     '''
   31     Function that allow plot the confusion matrix, in two formats, 
   32     percentage or total data.
   33     '''
   34     if Normalize:
   35         Confusion_Matrix = Matrix.astype('float') / Matrix.sum(axis=1)[:, np.newaxis]
   36     else:
   37         Confusion_Matrix = Matrix
   38
   39     plt.imshow(Confusion_Matrix, interpolation = 'nearest', cmap = Cmap)
   40     plt.title(Title)
   41     plt.colorbar()
   42     tick_marks = np.arange(len(Class_Name))
   43     plt.xticks(tick_marks, Class_Name, rotation = 45)
   44     plt.yticks(tick_marks, Class_Name)
   45
   46     fmt = '.2f' if Normalize else 'd' 
   47     thresh = Confusion_Matrix.max() / 2.
   48
   49     for i, j in itertools.product(range(Confusion_Matrix.shape[0]), range(Confusion_Matrix.shape[1])):
   50         plt.text(j, i, format(Confusion_Matrix[i, j], fmt), horizontalalignment = "center", color = "white" if Confusion_Matrix[i,j] > thresh else "black")
   51
   52     plt.tight_layout()
   53     plt.ylabel('Desired Output')
   54     plt.xlabel('Estimated Output')
   55
   56     plt.show()
```
**WARNING**
* To plot the confusion matrix, it's compute using the index of the ouput node that has more activation, and no directly the neural network output. in the lines 60 and 70 are created two vectors to save the index of the one hot value in the codification of each emotion category of the ouput testing set and the output of the neural network; the lines 104-106 determine the index descrived. The line 109 compute the confusion matrix.
```[PYTHON]
   60 Index_Y_Test = np.empty([len(Y_Test),1], dtype = np.float64)
   70 Index_Y_Pred = np.empty([len(Y_Test),1], dtype = np.float64)
  ...
  104 for i in range(len(Y_Test)):
  105     Index_Y_Pred[i] = np.argmax(Y_Pred[i,:])     
  106     Index_Y_Pred[i] = np.argmax(Y_Pred[i,:])
  ...
  109 Confusion_Matrix = confusion_matrix(Index_Y_Test,Index_Y_Pred)
```
The next image show the confusion matrix that represent the performance of the neural network designed by the author.
![confusion matrix](https://user-images.githubusercontent.com/31509775/32284284-bf32ebda-bef4-11e7-820e-b14aba8524b3.png)

Finally is saved the model created to be used in the differents tools developed: [RNA_Emotions_BodyPosture_Keras_Tensorflow.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Classify%20emotions/RNA_Emotions_BodyPosture_Keras_Tensorflow.py) and [Recognition_And_Learning_BodyLenguage_System.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project).

**WARNING**
* You can change the "version" of the model saved to don't rewrite the previous model saved using the line 121.
* Please make sure of the path that has the address of the neural network model be correct in the line 122 (don't change or delete the model name that are written after the last slash, if you do it, please check the path to load the model in the tools that use the model):
```[PYTHON]
  120 # Is saved the model of the neural network.
  121 Version = '1' # Version of the model created.
  122 RNA.save('...\Model_RNA_Recognition_Of_Emotions ' + Version)
```






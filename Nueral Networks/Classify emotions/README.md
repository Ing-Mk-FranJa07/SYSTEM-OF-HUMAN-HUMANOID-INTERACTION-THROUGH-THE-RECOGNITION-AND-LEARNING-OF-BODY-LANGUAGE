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

### Second step: Organize the Data Base.

The script [DataBase_Organize.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Classify%20emotions/DataBase_Organize.py) was wrote to organize the data base unifying alls .csv files and creating two new .csv files, the first of them has the 80% of the whole data used to train the neural network and the second file has the 20% of the data used to test the neural network.

Also, this script add the codification to each category, in the both files, used like output of the neural network.
D:\Tesis\Pictures
**WARNINGS**
* Please make sure of the path that has the address of the .csv files be correct in the follow lines (don't change or delete the files names that are wroten after the last slash):
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
 * Note that in the lines 117-119 are being changed information into the .csv files; if you're working with the [.csv files created by the autor](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator/Emotions%20DataBase), don't change or delet this lines. If you're working with new data created using the tool: [DataBaseCreatorHumanPosture.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator), these lines are not necessaries.
 ```[PYTHON]
     117    Data = Data_Train.replace("Opened", 1)
     118    Data = Data_Train.replace("Closed", 2)
     119    Data = Data_Train.replace("Unknow", 3)
 ```
### Third step: Neural Network structure.

The first part of the code in the script: [RNA_Emotions_BodyPosture_Keras_Tensorflow.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Classify%20emotions/RNA_Emotions_BodyPosture_Keras_Tensorflow.py) load the traininig and the testing data, creating the sets to train and test the neural network. 

**WARNING**
* Please make sure of the path that has the address of the .csv files be correct in the follow lines (don't change or delete the files names that are wroten after the last slash):
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
Aster that the Data has been loaded and schuffled in training and testing sets; is created the model of the neural network. The neural network has 23 inputs and two hidden layers, the first of them has the same numbers of nodes that the inputs, and the second hidden layer has 17 nodes; the both layers have a Relu activation function. The ouput layer has 6 nodes and its activation function is sigmoidal because each node just can take two values: (0, 1) The total output is an "one hot" vector that contains six values. 

The next image is a representation of the neural network designed.
![Ann classification problem](https://user-images.githubusercontent.com/31509775/32282186-e80f7e7a-beee-11e7-85a2-af58946356f5.PNG)








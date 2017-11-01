# Artificial Neural Network which to classify the emotions using data from body posture.

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

**WARNINGS**
* Please make sure of the path that get the ubication of the .csv files be correct in the follow lines:
```[PYTHON]
    Category_Train = open("...\DataSet_Organized\Emotions_Unified\Postures_Train " + str(File_Code) + ".csv", "w") 
```









'''
Created on 9/08/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

Script that allow unified all .csv files generated.
'''
# Are imported the necessary wrappers.
import pandas as pd
import csv

# Is created a list with the header of each column.
Header = ['Head Yaw','Head Roll','Head Pitch',
          'Right Shoulder Roll','Right Shoulder Pitch',
          'Right Elbow Yaw','Right Elbow Roll',
          'Right Wrist Yaw',
          'Right Hand',
          'Left Shoulder Roll','Left Shoulder Pitch',
          'Left Elbow Yaw','Left Elbow Roll',
          'Left Wrist Yaw',
          'Left Hand',
          'Waist Roll', 'Waist Pitch',
          'Right Hip Roll', 'Right Hip Pitch',
          'Right Knee Yaw', 'Right Knee Roll',
          'Left Hip Roll', 'Left Hip Pitch',
          'Left Knee Yaw', 'Left Knee Roll',
          'Emotion']
  
def UnifiedCategories(File_Category, File_Number, File_Code):
    '''
    Function that create two files to each category, one of them has the 80% 
    of the total data saved and the another has the 20% of the total data saved.
    '''
    # Are created the new files.
    Category_Train = open("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\Postures_Train " + str(File_Code) + ".csv", "w") 
    Category_Train_csv = csv.writer(Category_Train)
    
    Category_Test = open("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\Postures_Test " + str(File_Code) + ".csv", "w") 
    Category_Test_csv = csv.writer(Category_Test)
    
    # Is saved the header in both files.
    for M in range(0,len(Header)):                                              
        Category_Train.write(str(Header[M]))
        Category_Test.write(str(Header[M]))
        if M < len(Header)-1:
                Category_Train.write(",")
                Category_Test.write(",")

    Category_Train.write("\n")
    Category_Test.write("\n") 
    
    # The data are unified, creating a train and test data files.
    for F in range(1, File_Number):
            FileName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\ " + str(File_Category) + " " + str(F) + ".csv")
            File = pd.read_csv(FileName, header = 0)
              
            for L in range (1,int((File.shape[0]-2)*0.8)):
                Category_Train_csv.writerow(File.loc[L])
            
            for L in range (int((File.shape[0]-2)*0.8), File.shape[0]-2):
                Category_Test_csv.writerow(File.loc[L])
      
    Category_Train.close()
    Category_Test.close()
  
def UnifiedFiles():
    '''
    Function that create two files, one that has the whole data to train the 
    neural network, and the another that contain the whole data to test the 
    neural network.
    '''
    # Are created the two files.
    Postures_Train = open("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataSetPostures_Train.csv", "w") 
    Postures_Train_csv = csv.writer(Postures_Train)
    
    Postures_Test = open("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataSetPostures_Test.csv", "w") 
    Postures_Test_csv = csv.writer(Postures_Test)
      
    # Is saved the header in both files.
    for M in range(0,len(Header)):
        Postures_Train.write(str(Header[M]))
        Postures_Test.write(str(Header[M]))
        if M < len(Header)-1:
            Postures_Train.write(",")
            Postures_Test.write(",")
      
    Postures_Train.write("\n")
    Postures_Test.write("\n")
    
    # The data are unified, creating a train and test data files.
    for F in range(1,7):
            
            # Train data file.
            FileName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\Postures_Train " + str(F) + ".csv")
            File = pd.read_csv(FileName, header = 0)
               
            for L in range (0,len(File)):
                Postures_Train_csv.writerow(File.loc[L])
                
            # Test data file.
            FileName_Test = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\Postures_Test " + str(F) + ".csv")
            File_Test = pd.read_csv(FileName_Test, header = 0)
               
            for L in range (0,len(File_Test)):
                Postures_Test_csv.writerow(File_Test.loc[L])
       
    Postures_Train.close()
    Postures_Test.close()

def AddOutput(File_Type):
    '''
    Function that add the code to each emotion to the data files. This codification
    are used like output to the neural network.
    '''
    # Is loaded the file.
    Data = pd.read_csv("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataSetPostures_" + str(File_Type) + ".csv", header = 0)
    
    # Are added new columns to the output code.
    Data['Emotion Code 6'] = 0
    Data['Emotion Code 5'] = 0
    Data['Emotion Code 4'] = 0
    Data['Emotion Code 3'] = 0
    Data['Emotion Code 2'] = 0
    Data['Emotion Code 1'] = 0
    
    # Is changed the index of the data frame.
    Data.set_index("Emotion", inplace = True)
    
    # Are replaced the columns that has the output code with the correct code.
    Data.loc[["Happy"],["Emotion Code 1"]] = 1
    Data.loc[["Sad"],["Emotion Code 2"]] = 1
    Data.loc[["Angry"],["Emotion Code 3"]] = 1
    Data.loc[["Surprised"],["Emotion Code 4"]] = 1
    Data.loc[["Reflexive"],["Emotion Code 5"]] = 1
    Data.loc[["Normal"],["Emotion Code 6"]] = 1
    
    print(Data)
    
    # Is saved the data file.
    Data.to_csv("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataSetPostures__" + str(File_Type) + ".csv", header = True)

UnifiedCategories("Happy", 41, 1)
UnifiedCategories("Sad", 41, 2)
UnifiedCategories("Angry", 41, 3)
UnifiedCategories("Surprised", 41, 4)
UnifiedCategories("Reflexive", 41, 5)
UnifiedCategories("Normal", 31, 6)

UnifiedFiles()

AddOutput("Train")
AddOutput("Test")

print("Data base created!")

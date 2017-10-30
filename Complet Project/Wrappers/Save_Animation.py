'''
Created on 19/09/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

Wrapper that allow verify and save new motion animation sequences in 
.csv format files.
'''
# Are imported the necessary wrappers.
import time
import pandas as pd
import numpy as np

def VerifyAndSenSecuence(Animation, Pepper):
    '''
    Function that verify and sent to Pepper the motion animation sequence created.
    Also, get from Pepper its joint positions and save in a array the information.
    '''
    Pepper.InitPosition()                                                       # Send Pepper to initial position.  
    NewAnimation = np.empty([39, 16])                                   
    
    # --------------------------------------------------------------------
    # The animation matrix is divided in specific arrays.
    
    # Hip arrays.
    HipPitchList = Animation[:,14]; HipRollList = Animation[:,13]
    
    # Head array.
    HeadPitchList = Animation[:,0]
    
    # Right arm arrays.
    RShoulderPitchList = Animation[:,2]; RShoulderRollList = Animation[:,1]
    RElbowYawList = Animation[:,3]; RElbowRollList = Animation[:,4]
    RWristYawList = Animation[:,5]; RHandList = Animation[:,6]
    
    # Left arm arrays.
    LShoulderPitchList = Animation[:,8]; LShoulderRollList = Animation[:,7]
    LElbowYawList = Animation[:,9]; LElbowRollList = Animation[:,10]
    LWristYawList = Animation[:,11]; LHandList = Animation[:,12]          
    
    # --------------------------------------------------------------------
    # The current lines verify if the first angles are "None", if any of 
    # them is "None", is changed by the initial position angle of the 
    # specific joint.
    
    # Head.
    if HeadPitchList[0] == None:
        HeadPitchList[0] = -0.20000
    
    # Right arm.
    if RShoulderPitchList[0] == None:
        RShoulderPitchList[0] = 1.55968
    if RShoulderRollList[0] == None:
        RShoulderRollList[0] = -0.14272
    if RElbowRollList[0] == None:
        RElbowRollList[0] = 0.52254
    if RElbowYawList[0] == None:
        RElbowYawList[0] = 1.22826
    if RWristYawList[0] == None:
        RWristYawList[0] = 0.00050
    if RHandList[0] == None:
        RHandList[0] = 2
        
    # Left arm.    
    if LShoulderPitchList[0] == None:
        LShoulderPitchList[0] = 1.55968
    if LShoulderRollList[0] == None:
        LShoulderRollList[0] = 0.14272
    if LElbowRollList[0] == None:
        LElbowRollList[0] = -0.52254
    if LElbowYawList[0] == None:
        LElbowYawList[0] = -1.22826
    if LWristYawList[0] == None:
        LWristYawList[0] = -0.00050
    if LHandList[0] == None:
        LHandList[0] = 2
    # ---------------------------------------------------------------------
    # The current lines verify each angle searching for a "None", if it is 
    # found is changed for the previous correct angle value in each array, 
    # and then the angle is sent to Pepper.
    
    for M in range(0,len(HipRollList)):
        
        # Hip.
        Pepper.VerifyHipAngles('HipPitch', HipPitchList[M])
        Pepper.VerifyHipAngles('HipRoll', HipRollList[M])
        
        # Head.
        if HeadPitchList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if HeadPitchList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    HeadPitchList[M] = HeadPitchList[M-I]
            
            Pepper.VerifyHeadPitchAngles(HeadPitchList[M], 0)
        
        else:
            Pepper.VerifyHeadPitchAngles(HeadPitchList[M], 0)
          
        # Right shoulder.
        if RShoulderPitchList[M] == None and RShoulderRollList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if RShoulderPitchList[M-I] == None and RShoulderRollList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                   
                    RShoulderPitchList[M] = RShoulderPitchList[M-I]
                    RShoulderRollList[M] = RShoulderRollList[M-I]
           
            Pepper.VerifyShouldersAndWristsAngles('RShoulderPitch', RShoulderPitchList[M])
            Pepper.VerifyShouldersAndWristsAngles('RShoulderRoll', RShoulderRollList[M])
        
        else:
            Pepper.VerifyShouldersAndWristsAngles('RShoulderPitch', RShoulderPitchList[M])
            Pepper.VerifyShouldersAndWristsAngles('RShoulderRoll', RShoulderRollList[M])
        
        # Right elbow.
        if RElbowYawList[M] == None and RElbowRollList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if RElbowYawList[M-I] == None and RElbowRollList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    RElbowYawList[M] = RElbowYawList[M-I]
                    RElbowRollList[M] = RElbowRollList[M-I]
           
            Pepper.VerifyRElbowYawAngles(RElbowYawList[M])
            Pepper.VerifyRElbowRollAngles(RElbowRollList[M], RElbowYawList[M])
        
        else:
            Pepper.VerifyRElbowYawAngles(RElbowYawList[M])
            Pepper.VerifyRElbowRollAngles(RElbowRollList[M], RElbowYawList[M])
        
        # Right wrist.
        if RWristYawList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if RWristYawList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    RWristYawList[M] = RWristYawList[M-I]
            
            Pepper.VerifyShouldersAndWristsAngles('RWristYaw', RWristYawList[M])
        
        else:
            Pepper.VerifyShouldersAndWristsAngles('RWristYaw', RWristYawList[M])
        
        # Right hand.
        if RHandList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if RHandList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    RHandList[M] = RHandList[M-I]
            
            Pepper.SendToPepperHands('RHand', RHandList[M])
        
        else:
            Pepper.SendToPepperHands('RHand', RHandList[M])
        
        # Left shoulder.
        if LShoulderPitchList[M] == None and LShoulderRollList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if LShoulderPitchList[M-I] == None and LShoulderRollList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    LShoulderPitchList[M] = LShoulderPitchList[M-I]
                    LShoulderRollList[M] = LShoulderRollList[M-I]
            
            Pepper.VerifyShouldersAndWristsAngles('LShoulderPitch', LShoulderPitchList[M])
            Pepper.VerifyShouldersAndWristsAngles('LShoulderRoll', LShoulderRollList[M])
        
        else:
            Pepper.VerifyShouldersAndWristsAngles('LShoulderPitch', LShoulderPitchList[M])
            Pepper.VerifyShouldersAndWristsAngles('LShoulderRoll', LShoulderRollList[M])
        
        # Left elbow.
        if LElbowYawList[M] == None and LElbowRollList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if LElbowYawList[M-I] == None and LElbowRollList[M-I] == None:
                    I += 1
                else:
                    Flag = False
                    
                    LElbowYawList[M] = LElbowYawList[M-I]
                    LElbowRollList[M] = LElbowRollList[M-I]
            
            Pepper.VerifyLElbowYawAngles(LElbowYawList[M])
            Pepper.VerifyLElbowRollAngles(LElbowRollList[M], LElbowYawList[M])
        
        else:
            
            Pepper.VerifyLElbowYawAngles(LElbowYawList[M])
            Pepper.VerifyLElbowRollAngles(LElbowRollList[M], LElbowYawList[M])
        
        # Left wrist.
        if LWristYawList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if LWristYawList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    LWristYawList[M] = LWristYawList[M-I]
            
            Pepper.VerifyShouldersAndWristsAngles('LWristYaw', LWristYawList[M])
       
        else:
            Pepper.VerifyShouldersAndWristsAngles('LWristYaw', LWristYawList[M])
        
        # Left hand.
        if LHandList[M] == None:
            Flag = True
            I = 1
            while Flag == True:
                if LHandList[M-I] == None: 
                    I += 1
                else: 
                    Flag = False
                    
                    LHandList[M] = LHandList[M-I]
            
            Pepper.SendToPepperHands('LHand', LHandList[M])
            
        else:
            Pepper.SendToPepperHands('LHand', LHandList[M])
        
        time.sleep(0.15)
        
        # Are register the Pepper's joint orientations.
        
        # Hip orientation.
        NewAnimation[M, 0] = Pepper.GetToPepper('HipPitch'); NewAnimation[M, 1] = Pepper.GetToPepper('HipRoll')
        
        # Head orientation.
        NewAnimation[M, 2] = 0; NewAnimation[M, 3] = Pepper.GetToPepper('HeadPitch')
        
        # Right arm orientation.
        NewAnimation[M, 4] = Pepper.GetToPepper('RShoulderPitch'); NewAnimation[M, 5] = Pepper.GetToPepper('RShoulderRoll')
        NewAnimation[M, 6] = Pepper.GetToPepper('RElbowRoll'); NewAnimation[M, 7] = Pepper.GetToPepper('RElbowYaw')
        NewAnimation[M, 8] = Pepper.GetToPepper('RWristYaw')
        NewAnimation[M, 9] = Pepper.GetToPepper('RHand')
        
        # Left arm orientation.
        NewAnimation[M, 10] = Pepper.GetToPepper('LShoulderPitch'); NewAnimation[M, 11] = Pepper.GetToPepper('LShoulderRoll')
        NewAnimation[M, 12] = Pepper.GetToPepper('LElbowRoll'); NewAnimation[M, 13] = Pepper.GetToPepper('LElbowYaw')
        NewAnimation[M, 14] = Pepper.GetToPepper('LWristYaw')
        NewAnimation[M, 15] = Pepper.GetToPepper('LHand')
        
    return NewAnimation

def CreateFile(Animation):
    '''
    Function that create a .csv file.
    '''
    # Is created a header that has the names of each joint angle use to control Pepper.
    Header = ['Hip Pitch','Hip Roll',
              'Head Yaw','Head Pitch',
              'Right Shoulder Pitch','Right Shoulder Roll',
              'Right Elbow Yaw','Right Elbow Roll',
              'Right Wrist Yaw','Right Hand',
              'Left Shoulder Pitch','Left Shoulder Roll',
              'Left Elbow Yaw','Left Elbow Roll',
              'Left Wrist Yaw','Left Hand']
    
    # ------------------------------------------------------------------------
    # The current lines allow created a new file without overwrite the previous 
    # files, first are read all that created files with the name: 
    # "New_Animation_FileNumber.csv"; where FileNumber is an int.
    
    FileNumber = 1                                                              
    Continue = True
    
    while Continue:
        try:
            FileName = str("...\Data_And_RNA_Models\New_Animation_Data\New_Animation_" + str(FileNumber) + ".csv")
            File = pd.read_csv(FileName, header = None)
            
            FileNumber += 1
        except:
            Continue = False
            pass
    
    # ------------------------------------------------------------------------
    # Is created a new .csv file to save the joint angles.
        
    FileName = str("...\Data_And_RNA_Models\New_Animation_Data\New_Animation_" + str(FileNumber) + ".csv")
    File = open(FileName, 'w') 
     
    for M in range(0,len(Header)):
        File.write(str(Header[M]))
        if M < len(Header)-1:
            File.write(",")
    
    File.write("\n")
    for M in range(0,len(Animation)):
        
        # Hip angles.
        File.write(str(Animation[M, 0])); File.write(","); File.write(str(Animation[M, 1])); File.write(",");
        
        # Head angles. 
        File.write(str(Animation[M, 2])); File.write(","); File.write(str(Animation[M, 3])); File.write(","); 
        
        # Right arm angles.
        File.write(str(Animation[M, 4])); File.write(","); File.write(str(Animation[M, 5])); File.write(",");             
        File.write(str(Animation[M, 6])); File.write(","); File.write(str(Animation[M, 7])); File.write(",");             
        File.write(str(Animation[M, 8])); File.write(","); 
        File.write(str(Animation[M, 9])); File.write(","); 
        
        # Left arm angles.
        File.write(str(Animation[M, 10])); File.write(","); File.write(str(Animation[M, 11])); File.write(",");             
        File.write(str(Animation[M, 12])); File.write(","); File.write(str(Animation[M, 13])); File.write(",");             
        File.write(str(Animation[M, 14])); File.write(","); 
        File.write(str(Animation[M, 15]))
        
        File.write("\n")

    File.close()
            
            

'''
Created on 23/08/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This script allow the creation of .csv files that have motion sequences created
by the user using the Kinect Sensor, this sequences are recording in a limit time
and then are sending to be played by the Robot Pepper. 

If the user is agreeable with the performance, he can save the motion sequences
with a specific name that will be the name of the .csv file and the name of the 
motion sequences into the file.

Also is created a .avi file that save the video of the motion sequences developed
by the user.
'''
#coding: iso-8859-1

# The wrappers necessaries are imported.
from PyQt4 import uic, QtCore, QtGui                                            # Wrappers that allow used the GUI created.
from naoqi import ALProxy                                                       # Wrapper used to control the Robot Pepper.
from pykinect2 import PyKinectV2, PyKinectRuntime                               # Wrapper used to used the Kinect Sensor.
import pygame
import ctypes
import almath
import time
import cv2
import numpy as np
from cmath import pi
import transformations as tf                                                    # Wrapper used to handle matrices.
import pandas as pd

import sys 
if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# Function that show correctly the text in the GUI.
try:
    Encoding = QtGui.QApplication.UnicodeUTF8
    def Translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, Encoding)
except AttributeError:
    def Translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

# Is defined a vector that has the parameters to draw the skeleton.
SkeletonColors = [pygame.color.THECOLORS["green"],
                  pygame.color.THECOLORS["red"]]

# Is activated the tracking of the wrist.
ENABLE_WRIST = False

# Is initialized Pepper's parameters.
Speed_Arm_R = 1.0
Speed_Arm_L = 1.0
Speed_Head = 1.0
Speed_Hip = 1.0

JointsParams = {'RShoulderRoll':  {'position': 0, 'fractionMaxSpeed': 0.4 * Speed_Arm_R},
                'RShoulderPitch': {'position': 0, 'fractionMaxSpeed': 0.4 * Speed_Arm_R},
                'RElbowRoll':     {'position': 0, 'fractionMaxSpeed': 0.5 * Speed_Arm_R},
                'RElbowYaw':      {'position': 0, 'fractionMaxSpeed': 0.5 * Speed_Arm_R},
                'RWristYaw':      {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Arm_R},
                'RHand':          {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Arm_R},
                'LShoulderRoll':  {'position': 0, 'fractionMaxSpeed': 0.4 * Speed_Arm_L},
                'LShoulderPitch': {'position': 0, 'fractionMaxSpeed': 0.4 * Speed_Arm_L},
                'LElbowRoll':     {'position': 0, 'fractionMaxSpeed': 0.5 * Speed_Arm_L},
                'LElbowYaw':      {'position': 0, 'fractionMaxSpeed': 0.5 * Speed_Arm_L},
                'LWristYaw':      {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Arm_L},
                'LHand':          {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Arm_L},
                'HeadPitch':      {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Head},
                'HeadYaw':        {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Head},
                'HipRoll':        {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Hip},
                'HipPitch':       {'position': 0, 'fractionMaxSpeed': 0.3 * Speed_Hip}}

# Are created lists to save the angles of each joint that describe the body posture.
HeadYawList = list(); HeadRollList = list(); HeadPitchList = list()             # Lists for head angles.
RShoulderRollList = list(); RShoulderPitchList = list()                         # Lists for right shoulder angles.
RElbowYawList = list(); RElbowRollList = list()                                 # Lists for right elbow angles.
RWristYawList = list()                                                          # List for right wrist angles.
RHandList = list()                                                              # List for right hand state.
LShoulderRollList = list(); LShoulderPitchList = list()                         # Lists for left shoulder angles.
LElbowYawList = list(); LElbowRollList = list()                                 # Lists for left elbow angles.
LWristYawList = list()                                                          # List for left wrist angles.
LHandList = list()                                                              # List for left hand state.
HipRollList = list(); HipPitchList = list()                                     # Lists for hip angles.

# Are created lists to save the angles of each joint that describe the Robot Pepper posture.
HeadPitchPepperList = list(); HeadYawPepperList = list()                        # Lists for Pepper head angles.
RShoulderPitchPepperList = list(); RShoulderRollPepperList = list ()            # Lists for Pepper right shoulder angles.
RElbowRollPepperList = list(); RElbowYawPepperList = list()                     # Lists for Pepper right elbow angles.
RWristYawPepperList = list();                                                   # List for Pepper right wrist angles.
RHandPepperList = list()                                                        # List for Pepper right hand state.
LShoulderPitchPepperList = list(); LShoulderRollPepperList = list ()            # Lists for Pepper left shoulder angles.
LElbowRollPepperList = list(); LElbowYawPepperList = list()                     # Lists for Pepper left elbow angles.         
LWristYawPepperList = list();                                                   # List for Pepper left wrist angles.
LHandPepperList = list()                                                        # List for Pepper left hand state.
HipRollPepperList = list(); HipPitchPepperList = list()                         # Lists for Pepper hip angles.    

# Is created a list with the header of each column used to create a .csv file.
Header = ['Animation',
          'Hip Pitch','Hip Roll',
          'Head Yaw','Head Pitch',
          'Right Shoulder Pitch','Right Shoulder Roll',
          'Right Elbow Yaw','Right Elbow Roll',
          'Right Wrist Yaw','Right Hand',
          'Left Shoulder Pitch','Left Shoulder Roll',
          'Left Elbow Yaw','Left Elbow Roll',
          'Left Wrist Yaw','Left Hand']

def add_angles(alpha, beta):
    '''
    This function return the add of two angles, getting the result
    into the interval [-pi,pi]
    '''
    Add = alpha + beta

    while np.abs(Add) > (2*pi):
        if Add > 0:
            Add -= 2*pi
        else:
            Add += 2*pi 

    if Add > pi:
        return Add - (2*pi) 
    elif Add < -pi:
        return Add + (2*pi) 
    else:
        return Add 

class GUI(QtGui.QWidget):
    '''
    Main class that allow the operation of the system using the GUI created, to
    show the image gotten using the Kinect Sensor, and compute the skeleton 
    tracking to generate motion sequence and send them to Robot Pepper.
    
    This class also allow save and load motion sequences.
    '''
    def __init__(self):
        '''
        Function that initialize the GUI parameters and connect them with the
        corresponding functions.
        '''
        super(GUI, self).__init__()
        
        # Is loaded the file that get the GUI.  
        self.MyGUI = uic.loadUi('...\AnimationDataBaseCreatorGUI.ui', self)
        
        # Are associated the different options that have the GUI with its corresponding functions.
        self.connect(self.Btn_Conect_To_Pepper, QtCore.SIGNAL("clicked()"), self.InitConection)
        self.connect(self.Btn_Start_End_Tracking, QtCore.SIGNAL("clicked()"), self.InitKinect)
        self.connect(self.Btn_Recording, QtCore.SIGNAL("clicked()"), self.Recording)
        self.connect(self.Btn_Send_Secuence_To_Pepper, QtCore.SIGNAL("clicked()"), self.SendSecuence)
        self.connect(self.Btn_Save, QtCore.SIGNAL("clicked()"), self.CreateFileCSV)
        self.connect(self.Btn_Load, QtCore.SIGNAL("clicked()"), self.LoadFileCSV)
        self.connect(self.Btn_Init_Position, QtCore.SIGNAL("clicked()"), self.InitPosition)
        self.connect(self.Btn_Close, QtCore.SIGNAL("clicked()"), self.EndProgram)     
        
        # Are created timers to have interruptions that allow execute functions.
        self.Timer = QtCore.QTimer(self.MyGUI)                                  # Timer that control the image tracking
        self.Timer.timeout.connect(self.ShowFrame)                              # The timer active the Tracking function.
        
        self.Timer1 = QtCore.QTimer(self.MyGUI)                                 # Timer that control the playing of .avi file.
        self.Timer1.timeout.connect(self.ShowVideo)                             # The timer active the playing of .avi file.

        self.Timer2 = QtCore.QTimer(self.MyGUI)                                 # Timer that control the playing of the motion sequences in the Robot Pepper.
        self.Timer2.timeout.connect(self.ShowAnimation)                         # The timer activate the playing of the motion sequences in the Robot Pepper.
        
        self.MyGUI.show()
        
    def InitConection(self):
        '''
        Function that get the Ip and Port number to start the connection with Pepper.
        '''
        try:
            # Are gotten from the GUI the IP and Port to connect with Robor Pepper.
            self.Ip = str(self.Text_IP.text())
            self.Port = int(self.Text_Port.text())
    
            # Are created the objects to control the Robot Pepper.
            self.Motion = ALProxy('ALMotion', self.Ip, self.Port)
            self.Posture = ALProxy('ALRobotPosture', self.Ip, self.Port)
                    
            # Is "woken up" the Robot Pepper.
            self.Motion.wakeUp()
            self.Posture.goToPosture('Stand', 0.75)
            

            self.Frame_Conection.setEnabled(False)                              # Disabled the connection options with Robot Pepper.
            self.Btn_Start_End_Tracking.setEnabled(True)                        # Enabled the option to connect with the Kinect Sensor.    
            self.Btn_Load.setEnabled(True)                                      # Enabled the load of the motion sequences saved.
            self.Btn_Init_Position.setEnabled(True)                             # Enabled the option to send the Robot Pepper to init position.
            self.Text_File_Name.setEnabled(True)                                # Enabled the text box to put the name of .avi and .csv files.
            
            # Is created a pixel map to put in the GUI the image get with the Kinect V2 Camera.
            self.pixmap = QtGui.QPixmap()

            self.PlayVideo = 0                                                  # Flag to start or end the playing of the .avi file.
            self.StartEnd = 1                                                   # Flag to start or Stop Kinect Sensor.
            
        except (RuntimeError, ValueError):                                      # Error alerts.
            if str(self.Text_Port.text()) == '':
                self.Text_Port.setText("Empty!")
            else:
                self.Text_Port.setText("Wrong!")
            if str(self.Text_IP.text()) == '':
                self.Text_IP.setText("Empty!")
                
            pass
    
    def InitKinect(self):
        '''
        Function that start or Stop Kinect Sensor.
        '''
        # Connect Kinect.
        if self.StartEnd == 1:
            # Are configured the parameters to use the Kinect Sensor.
            pygame.init()
            self.Clock = pygame.time.Clock()
            self.Kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
            self.FrameSurface = pygame.Surface((self.Kinect.color_frame_desc.Width, self.Kinect.color_frame_desc.Height), 0, 32)
    
            self.Bodies = None                                                  # Variable used to saved the bodies detected.
            
            self.Btn_Recording.setEnabled(True)                                 # Enabled the recording option.
            self.Btn_Start_End_Tracking.setText("Disconnect Kinect")            # Change the text in the GUI button.
            
            self.StartEnd = 0                                                   # Change the flag's value to Stop Kinect Sensor.
                    
        else:
            try:
                self.StartEnd = 1                                               # Change the flag's value to start the Kinect Sensor.
                
                self.Btn_Recording.setEnabled(False)                            # Disabled the recording option.
                self.Btn_Send_Secuence_To_Pepper.setEnabled(False)              # Enabled the option to send the motion sequences to Robot Pepper.
                self.Btn_Recording.setText("REC")                               # Change the text in the GUI button.
                self.Btn_Start_End_Tracking.setText("Connect Kinect")           # Change the text in the GUI button.
                
                self.Kinect.close()                                             # Stop Kinect Sensor.
                pygame.quit()
                
                self.VideoFile.release()                                        # End the creation of the .avi file.
            except:
                pass

    def Recording(self):
        '''
        Function that activate the recording of the motion sequence.
        '''
        self.PlayVideo = 0                                                      # Change the flag's value to end the playing of the .avi file.
        
        self.Btn_Save.setEnabled(False)                                         # Disabled the option to save a new motion sequence in a .csv file.
        self.Btn_Load.setEnabled(False)                                         # Disabled the option to load a motion sequence saved.
        self.Btn_Send_Secuence_To_Pepper.setEnabled(False)                      # Disabled the option to send the motion sequences to Robot Pepper.    
        self.Btn_Init_Position.setEnabled(False)                                # Disabled the option to send the Robot Pepper to the init position.
        self.Btn_Recording.setEnabled(False)                                    # Disabled the recording option.
        self.Text_File_Name.setEnabled(False)                                   # Disabled the text box to put the name of the .avi and .csv files.
        self.Btn_Recording.setText("Recording")                                 # Change the text in the GUI button.                              
        
        # Is saved the motion sequences name.
        self.FileName = str(self.Text_File_Name.text())
        VideoFilePath =  str("...\Motion_Sequences\ " + self.FileName + ".avi")
        
        # Is definite the format of the video file.
        VideoFileType = cv2.VideoWriter_fourcc('M','J','P','G')
        
        # Is create the .avi file.
        self.VideoFile = cv2.VideoWriter(VideoFilePath, VideoFileType, 10.0, (1080,630))

        # Are imported like global the lists used to save the information.  
        global HeadPitchList, HeadYawList
         
        global RShoulderPitchList, RShoulderRollList
        global RElbowRollList, RElbowYawList
        global RWristYawList, RHandList
         
        global LShoulderPitchList, LShoulderRollList
        global LElbowRollList, LElbowYawList
        global LWristYawList, LHandList
 
        global HipPitchList, HipRollList

        # Are cleared the lists used to save the information.
        HeadPitchList= []; HeadYawList= []
         
        RShoulderPitchList= []; RShoulderRollList= []
        RElbowRollList= []; RElbowYawList= []
        RWristYawList= []; RHandList= []
         
        LShoulderPitchList= []; LShoulderRollList= []
        LElbowRollList= []; LElbowYawList= []
        LWristYawList= []; LHandList= []
         
        HipPitchList= []; HipRollList= []
        
        # Are cleared the lists used to save the Robot Pepper posture.
        del(HeadPitchPepperList[::]); del(HeadYawPepperList[::])
        
        del(RShoulderPitchPepperList[::]); del(RShoulderRollPepperList[::])
        del(RElbowRollPepperList[::]); del(RElbowYawPepperList[::])
        del(RWristYawPepperList[::]); del(RHandPepperList[::])
        
        del(LShoulderPitchPepperList[::]); del(LShoulderRollPepperList[::])
        del(LElbowRollPepperList[::]); del(LElbowYawPepperList[::])
        del(LWristYawPepperList[::]); del(LHandPepperList[::])
        
        del(HipPitchPepperList[::]); del(HipRollPepperList[::])
        
        time.sleep(2)                                                           # Delay used to leave the user prepare him.
        
        self.Segundos = 0                                                       # Counter that limit the recording.    
        
        self.Timer.start(1)                                                     # Activate the interruption that get the skeleton tracking.
            
    def SendSecuence(self):
        '''
        Function that send the motion sequences to Robot Pepper.
        '''      
        self.InitPosition()                                                     # Send Robot Pepper to initial position.  
        
        # The current lines verify if the first angles are "None", if any of 
        # them is "None", is changed by the initial position angle of the 
        # specific joint.if HeadPitchList[0] == None:
        
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
            
        # The current lines verify each angle searching for a "None", if it is 
        # found is changed for the previous correct angle value in each array, 
        # and then the angle is sent to Pepper.
        
        for M in range(0,len(HipRollList)):
          
            # Hip.
            self.VerifyHipAngles('HipPitch', HipPitchList[M])
            self.VerifyHipAngles('HipRoll', HipRollList[M])
            
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
                
                self.VerifyHeadPitchAngles(HeadPitchList[M], 0)
            
            else:
                self.VerifyHeadPitchAngles(HeadPitchList[M], 0)
              
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
               
                self.VerifyShouldersAndWristsAngles('RShoulderPitch', RShoulderPitchList[M])
                self.VerifyShouldersAndWristsAngles('RShoulderRoll', RShoulderRollList[M])
            
            else:
                self.VerifyShouldersAndWristsAngles('RShoulderPitch', RShoulderPitchList[M])
                self.VerifyShouldersAndWristsAngles('RShoulderRoll', RShoulderRollList[M])
            
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
               
                self.VerifyRElbowYawAngles(RElbowYawList[M])
                self.VerifyRElbowRollAngles(RElbowRollList[M], RElbowYawList[M])
            
            else:
                self.VerifyRElbowYawAngles(RElbowYawList[M])
                self.VerifyRElbowRollAngles(RElbowRollList[M], RElbowYawList[M])
            
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
                
                self.VerifyShouldersAndWristsAngles('RWristYaw', RWristYawList[M])
            
            else:
                self.VerifyShouldersAndWristsAngles('RWristYaw', RWristYawList[M])
            
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
                
                self.SendToPepperHands('RHand', RHandList[M])
            
            else:
                self.SendToPepperHands('RHand', RHandList[M])
            
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
                
                self.VerifyShouldersAndWristsAngles('LShoulderPitch', LShoulderPitchList[M])
                self.VerifyShouldersAndWristsAngles('LShoulderRoll', LShoulderRollList[M])
            
            else:
                self.VerifyShouldersAndWristsAngles('LShoulderPitch', LShoulderPitchList[M])
                self.VerifyShouldersAndWristsAngles('LShoulderRoll', LShoulderRollList[M])
            
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
                
                self.VerifyLElbowYawAngles(LElbowYawList[M])
                self.VerifyLElbowRollAngles(LElbowRollList[M], LElbowYawList[M])
            
            else:
                
                self.VerifyLElbowYawAngles(LElbowYawList[M])
                self.VerifyLElbowRollAngles(LElbowRollList[M], LElbowYawList[M])
            
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
                
                self.VerifyShouldersAndWristsAngles('LWristYaw', LWristYawList[M])
           
            else:
                self.VerifyShouldersAndWristsAngles('LWristYaw', LWristYawList[M])
            
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
                
                self.SendToPepperHands('LHand', LHandList[M])
                
            else:
                self.SendToPepperHands('LHand', LHandList[M])
                
            time.sleep(0.13)
            
            # Are register the Pepper's joint orientations.
            
            # Hip orientation.
            HipPitchPepperList.append(self.GetToPepper('HipPitch')); HipRollPepperList.append(self.GetToPepper('HipRoll'))
            
            # Head orientation.
            HeadYawPepperList.append(self.GetToPepper('HeadYaw')); HeadPitchPepperList.append(self.GetToPepper('HeadPitch')) 
            
            # Right arm orientation.
            RShoulderPitchPepperList.append(self.GetToPepper('RShoulderPitch')); RShoulderRollPepperList.append(self.GetToPepper('RShoulderRoll'))
            RElbowRollPepperList.append(self.GetToPepper('RElbowRoll')); RElbowYawPepperList.append(self.GetToPepper('RElbowYaw'))
            RWristYawPepperList.append(self.GetToPepper('RWristYaw'))
            RHandPepperList.append(self.GetToPepper('RHand'))
            
            # Left arm orientation.
            LShoulderPitchPepperList.append(self.GetToPepper('LShoulderPitch')); LShoulderRollPepperList.append(self.GetToPepper('LShoulderRoll'))
            LElbowRollPepperList.append(self.GetToPepper('LElbowRoll')); LElbowYawPepperList.append(self.GetToPepper('LElbowYaw'))
            LWristYawPepperList.append(self.GetToPepper('LWristYaw'))
            LHandPepperList.append(self.GetToPepper('LHand'))
        
        self.Btn_Save.setEnabled(True)                                          # Enabled the option to save the new motion sequence in the .csv file.
        self.Btn_Init_Position.setEnabled(True)                                 # Enabled the option to send Robot Pepper to init position.
        
    def CreateFileCSV(self):
        '''
        Function that allow create a .csv file to save the motion sequence.
        '''
        # Is saved the name of the motion sequence.
        AnimationName = str(self.Text_Animation.text())
        FileName = str(".../Motion_Sequences\ " + self.FileName + ".csv")
        
        # Is created the .csv file.
        File = open(FileName, 'w')                                              
         
        # Is saved the header in the .csv file.
        for M in range(0,len(Header)):
            File.write(str(Header[M]))
            if M < len(Header)-1:
                File.write(",")
        
        File.write("\n")
        
        # Are saved in the .csv file the motion sequence.
        for M in range(0,len(HipPitchList)):
             
            File.write(AnimationName); File.write(",")
             
            File.write(str(HipPitchPepperList[M])); File.write(","); File.write(str(HipRollPepperList[M])); File.write(",");
             
            File.write(str(HeadYawPepperList[M])); File.write(","); File.write(str(HeadPitchPepperList[M])); File.write(","); 
             
            File.write(str(RShoulderPitchPepperList[M])); File.write(","); File.write(str(RShoulderRollPepperList[M])); File.write(",");             
            File.write(str(RElbowYawPepperList[M])); File.write(","); File.write(str(RElbowRollPepperList[M])); File.write(",");             
            File.write(str(RWristYawPepperList[M])); File.write(","); 
            File.write(str(RHandPepperList[M])); File.write(","); 
 
            File.write(str(LShoulderPitchPepperList[M])); File.write(","); File.write(str(LShoulderRollPepperList[M])); File.write(",");             
            File.write(str(LElbowYawPepperList[M])); File.write(","); File.write(str(LElbowRollPepperList[M])); File.write(",");             
            File.write(str(LWristYawPepperList[M])); File.write(","); 
            File.write(str(LHandPepperList[M]))
             
            File.write("\n")
            
        File.close()                                                            # Close the .csv file.
        
        self.Text_File_Name.setEnabled(True)                                    # Enabled the text box to put the name of the .avi and .csv files.    
        self.Text_Animation.setEnabled(False)                                   # Disabled the text box to put the name of the motion sequence.
        self.Text_Animation.setText(" ")                                        # Clear the text in the text box used to name the motion sequence.
        self.Btn_Save.setEnabled(False)                                         # Disabled the option to save a new .csv file.
        self.Btn_Load.setEnabled(True)                                          # Enabled the option to load a motion sequences saved.
        self.Btn_Recording.setEnabled(True)                                     # Enabled the recording option.

    def LoadFileCSV(self):
        '''
        Function that allow load a motion sequence saved.
        '''
        self.Text_File_Name.setEnabled(False)                                   # Disabled the text box to put the name of the .avi and .csv files.
        self.Text_Animation.setEnabled(False)                                   # Disabled the text box to put the name of the motion sequence.
        self.Btn_Load.setEnabled(False)                                         # Disabled the option to load a motion sequence saved.
        self.Btn_Recording.setEnabled(False)                                    # Disabled the recording option.

        try:                                                                    # Stop Kinect Sensor.
            self.Kinect.close()
            pygame.quit()
        except AttributeError:
            pass
        
        # Are imported like global the lists used to save the information.  
        global HeadPitchList, HeadYawList
         
        global RShoulderPitchList, RShoulderRollList
        global RElbowRollList, RElbowYawList
        global RWristYawList, RHandList
         
        global LShoulderPitchList, LShoulderRollList
        global LElbowRollList, LElbowYawList
        global LWristYawList, LHandList
 
        global HipPitchList, HipRollList

        # Are cleared the lists used to save the information.
        HeadPitchList= []; HeadYawList= []
         
        RShoulderPitchList= []; RShoulderRollList= []
        RElbowRollList= []; RElbowYawList= []
        RWristYawList= []; RHandList= []
         
        LShoulderPitchList= []; LShoulderRollList= []
        LElbowRollList= []; LElbowYawList= []
        LWristYawList= []; LHandList= []
         
        HipPitchList= []; HipRollList= []
        
        # Is loaded the .avi and .csv files.
        try:
            # .csv file. 
            FileName = str("...\Motion_Sequences\ " + str(self.Text_File_Name.text()) + ".csv")
            File = pd.read_csv(FileName, header = 0)
            
            # .av file.
            self.VideoName = str("...\Motions_Sequences\ " + str(self.Text_File_Name.text()) + ".avi")
            self.Video = cv2.VideoCapture(self.VideoName)
            
            PlayVideo = 1                                                       # Check the correct load.    
            
        except:

            try:
                # .csv file 
                FileName = str("...\Motion_Sequences_Generate_By_RNA\ " + str(self.Text_File_Name.text()) + ".csv")
                File = pd.read_csv(FileName, header = None)
                
                PlayVideo = 2                                                   # Check the partial load.
            
            except:
                PlayVideo = 0                                                   # Check the incorrect load.

            pass
        
        pass
        
        # If the .avi and .csv files are loaded correctly, the motion sequence is
        # saved using the lists created for each angle to be played by the Robot 
        # Pepper, and also is played the video saved in the .avi file.
        #
        # If just the .csv file are loaded correctly, or just exist a .csv file,
        # the motion sequence is saved using the lists created for each angle 
        # to be played by the Robot Pepper.
        #
        # If the both files are not loaded correctly, the user is informed to 
        # put a correct name to load the files.   
        if PlayVideo == 1:
            # Hip angles.
            HipPitchList = File['Hip Pitch']; HipRollList = File['Hip Roll']
            
            # Head angles.
            HeadPitchList = File['Head Pitch']; HeadYawList = File['Head Yaw']
            
            # Right arm angles.
            RShoulderPitchList = File['Right Shoulder Pitch']; RShoulderRollList = File['Right Shoulder Roll']
            RElbowYawList = File['Right Elbow Yaw']; RElbowRollList = File['Right Elbow Roll']
            RWristYawList = File['Right Wrist Yaw']; RHandList = File['Right Hand']
            
            # Left arm angles.
            LShoulderPitchList = File['Left Shoulder Pitch']; LShoulderRollList = File['Left Shoulder Roll']
            LElbowYawList = File['Left Elbow Yaw']; LElbowRollList = File['Left Elbow Roll']
            LWristYawList = File['Left Wrist Yaw']; LHandList = File['Left Hand']
            
            self.M = 0                                                          # Counter used to play the .avi file.
            
            self.InitPosition()                                                 # Send Robot Pepper to init position.
            
            self.Timer1.start(100)                                              # Activate the interruption that show the .avi frames in the GUI, and play the motion sequence.
        
        elif PlayVideo == 2: 

            # Hip angles.
            HipPitchList = File.ix[:,0]; HipRollList = File.ix[:,1]
            
            # Head angles.
            HeadPitchList = File.ix[:,2]; HeadYawList = File.ix[:,3]
            
            # Right arm angles.
            RShoulderPitchList = File.ix[:,4]; RShoulderRollList = File.ix[:,5]
            RElbowYawList = File.ix[:,6]; RElbowRollList = File.ix[:,7]
            RWristYawList = File.ix[:,8]; RHandList = File.ix[:,9]
            
            # Left arm angles.
            LShoulderPitchList = File.ix[:,10]; LShoulderRollList = File.ix[:,11]
            LElbowYawList = File.ix[:,12]; LElbowRollList = File.ix[:,13]
            LWristYawList = File.ix[:,14]; LHandList = File.ix[:,15]

            self.M = 0                                                          # Counter used to play the .avi file.
            
            self.InitPosition()                                                 # Send Robot Pepper to init position.
            
            self.Timer2.start(100)                                              # Activate the interruption that play the motion sequences.
            
        else:
            self.Text_File_Name.setText("Enter a valid name")                   # The user is informed that the .avi and .csv files names are incorrect.
        
        self.Text_File_Name.setEnabled(True)                                    # Enabled the text box to put the name of the .avi and .csv files.
        self.Btn_Load.setEnabled(True)                                          # Enabled the option to load a motion sequence saved.
        self.Btn_Recording.setEnabled(True)                                     # Enabled the recording option.         

    def ShowFrame(self):
        '''
        Main function that allow the skeleton tracking and get the body posture
        to save the joint angles and then create the .avi and .cvs files.
        '''
        # Se actualiza el contrador de segundos.
        self.Segundos += 1
        
        # Deactive the timer in the limit time.
        if self.Segundos == 40:
            self.Timer.stop()                                                   # Deactivate the interruption that get the skeleton tracking.        
            
            try:                                                                # End the creation of the .avi file.
                self.VideoFile.release()
            except AttributeError:
                pass

            self.Text_File_Name.setEnabled(True)                                # Enabled the text box to put the name of the .avi and .csv files.
            self.Text_Animation.setEnabled(True)                                # Enabled the text box to put the name of the motion sequence.    
            self.Btn_Load.setEnabled(True)                                      # Enabled the option to load a motion sequence loaded.
            self.Btn_Recording.setEnabled(True)                                 # Enabled the recording option.
            self.Btn_Recording.setText("REC")                                   # Change the text in the GUI button.
            self.Btn_Send_Secuence_To_Pepper.setEnabled(True)                   # Enabled the option to send the motion sequences to Robot Pepper.
            
            return
        
        # Is reshape the Kinect's image in the format that allow cv2.
        Image = pygame.surfarray.array3d(self.FrameSurface)                     # Pygame's surface is converted in an array. 
        Image = np.rollaxis(Image, 0, 2)                                        # Reform the structure of the array.
         
        # Is reformat and reshape the Image to be put in the GUI.
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        Image = cv2.resize(Image,(1080,630), interpolation = cv2.INTER_CUBIC)
    
        if self.Kinect.has_new_color_frame():
            Frame = self.Kinect.get_last_color_frame()                          # Save the image get with the Kinect V2 Camera.
            self.DrawColorFrame(Frame, self.FrameSurface)   
            Frame = None
            
            if self.Kinect.has_new_body_frame(): 
                self.Bodies = self.Kinect.get_last_body_frame()                 # Save the bodies present in the image.

                if self.Bodies is not None:
                    Body = self.SelectNearestBody()                             # Select the nearest body.

                    if Body is not None:                                        # Draw and compute the body's joint angles.
                        Joints = Body.joints
                        JointsPoints = self.Kinect.body_joints_to_color_space(Joints)
                        Orientations = Body.joint_orientations 
                        self.DrawBody(Joints, JointsPoints, SkeletonColors)
                        self.Angles(Joints, Orientations, Body)              
        
        # Se crea un archivo de video .avi
        self.VideoFile.write(Image)                                             # Save a new frame in the .avi file.
        
        # Create a pixel map from the Kinect's image.
        Image = QtGui.QImage(self.Image, 1080, 630, 3240 , QtGui.QImage.Format_RGB888)
        self.pixmap.convertFromImage(Image.rgbSwapped())                        # Specify a RGB format.
        self.KinectFrame.setPixmap(self.pixmap)                                 # Show the Kinect's image in the GUI.
        
        self.Clock.tick(60)

    def ShowVideo(self):
        '''
        Function that allow play the video and the motion sequence saved in the
        .avi and .csv files.
        '''
        Image = self.Video.read()[1]                                            # Load a frame from the .avi file.    
                    
        # Create a pixel map from the Kinect's image.
        Image = QtGui.QImage(Image, 1080, 630, 3240 , QtGui.QImage.Format_RGB888)
        self.pixmap.convertFromImage(Image.rgbSwapped())                        
        self.KinectImage.setPixmap(self.pixmap)                                 # Show the Kinect's image in the GUI.
        
        # Send the motion sequence to Pepper.
        self.VerifyHipAngles('HipPitch', HipPitchList[self.M])
        self.VerifyHipAngles('HipRoll', HipRollList[self.M])

        if HipPitchList[self.M] is not 0:
            self.VerifyHeadYawAngles(HeadYawList[self.M])
        self.VerifyHeadPitchAngles(HeadPitchList[self.M], HeadYawList[self.M])

        self.VerifyShouldersAndWristsAngles('RShoulderPitch', RShoulderPitchList[self.M])
        self.VerifyShouldersAndWristsAngles('RShoulderRoll', RShoulderRollList[self.M])

        self.VerifyRElbowYawAngles(RElbowYawList[self.M])
        self.VerifyRElbowRollAngles(RElbowRollList[self.M], RElbowYawList[self.M])
        
        self.VerifyShouldersAndWristsAngles('RWristYaw', RWristYawList[self.M])

        self.Motion.setAngles('RHand', RHandList[self.M], JointsParams['RHand']['fractionMaxSpeed'])

        self.VerifyShouldersAndWristsAngles('LShoulderPitch', LShoulderPitchList[self.M])
        self.VerifyShouldersAndWristsAngles('LShoulderRoll', LShoulderRollList[self.M])

        self.VerifyLElbowYawAngles(LElbowYawList[self.M])
        self.VerifyLElbowRollAngles(LElbowRollList[self.M], LElbowYawList[self.M])

        self.VerifyShouldersAndWristsAngles('LWristYaw', LWristYawList[self.M])
    
        self.Motion.setAngles('LHand', LHandList[self.M], JointsParams['LHand']['fractionMaxSpeed'])
       
        self.M += 1
        
        # End the playing of the .avi and .csv file.
        if self.M == len(HipPitchList)-1:
            self.Timer1.stop()
            self.Video.release()
    
    def ShowAnimation(self):
        '''
        Function that allow play the motion sequecen saved in the .csv file.
        '''
        # Send the motion sequence to Pepper.
        self.VerifyHipAngles('HipPitch', HipPitchList[self.M])
        self.VerifyHipAngles('HipRoll', HipRollList[self.M])

        if HipPitchList[self.M] is not 0:
            self.VerifyHeadYawAngles(HeadYawList[self.M])
        self.VerifyHeadPitchAngles(HeadPitchList[self.M], HeadYawList[self.M])

        self.VerifyShouldersAndWristsAngles('RShoulderPitch', RShoulderPitchList[self.M])
        self.VerifyShouldersAndWristsAngles('RShoulderRoll', RShoulderRollList[self.M])

        self.VerifyRElbowYawAngles(RElbowYawList[self.M])
        self.VerifyRElbowRollAngles(RElbowRollList[self.M], RElbowYawList[self.M])
        
        self.VerifyShouldersAndWristsAngles('RWristYaw', RWristYawList[self.M])

        self.Motion.setAngles('RHand', RHandList[self.M], JointsParams['RHand']['fractionMaxSpeed'])

        self.VerifyShouldersAndWristsAngles('LShoulderPitch', LShoulderPitchList[self.M])
        self.VerifyShouldersAndWristsAngles('LShoulderRoll', LShoulderRollList[self.M])

        self.VerifyLElbowYawAngles(LElbowYawList[self.M])
        self.VerifyLElbowRollAngles(LElbowRollList[self.M], LElbowYawList[self.M])

        self.VerifyShouldersAndWristsAngles('LWristYaw', LWristYawList[self.M])
    
        self.Motion.setAngles('LHand', LHandList[self.M], JointsParams['LHand']['fractionMaxSpeed'])
        
        self.M += 1
        
        # End the playing of the .csv file.
        if self.M == len(HipPitchList)-1:
            self.Timer2.stop()

    def DrawColorFrame(self, Frame, TargetSurface):
        '''
        Function that decode the image get with the Kinect Sensor and 
        convert it in a compatible color image puting it in a pygame's surface.
        '''
        TargetSurface.lock()
        address = self.Kinect.surface_as_array(TargetSurface.get_buffer())      # Function that get the color image and save the frame.
        ctypes.memmove(address, Frame.ctypes.data, Frame.size)                  # C function that return a compatible color image.
        del address
        TargetSurface.unlock()

    def SelectNearestBody(self):
        '''
        Function that determined the nearest body in the image get with the 
        Kinect Sensor and return the information of that body.
        '''
        NearestBody = None
        NearestDistance = float('inf')
     
        for i in range(0, self.Kinect.max_body_count):
            Body = self.Bodies.bodies[i] 
            if not Body.is_tracked: 
                continue
            
            Spine = Body.joints[PyKinectV2.JointType_SpineBase].Position        # Spine coordinates.
            Distance = np.sqrt((Spine.x**2)+(Spine.y**2)+(Spine.z**2))          # Compute the eucledian distance of the body to the Kinect Sensor.

            if Distance < NearestDistance:
                NearestDistance = Distance
                NearestBody = Body
                
        return NearestBody 

    def DrawBody(self, Joints, JointsPoints, Color):
        '''
        Function that send the parameters get with the Kinect Sensor
        to draw the skeleton.
        '''
        # Torso
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft)
    
        # Right Arm
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);

    def Angles(self, Joints, Orientations, Body):
        '''
        Function that calculated the joint's angles of the skeleton get with the
        Kinect Sensor, getting the quaternion matrix and then calculating the
        eulerian angles.
        
        If the skeleton tracking is not correct, complete or parcial, the 
        corresponding angles are define like "None"; after that the angle are 
        verified, they are saved in the corresponding lists.
        '''  
        # Hip angles.
        if (Joints[PyKinectV2.JointType_SpineShoulder].TrackingState == 2) and (Joints[PyKinectV2.JointType_SpineBase].TrackingState == 2):
            ChestQuat = self.Quaternion(Orientations, PyKinectV2.JointType_SpineShoulder) 
            HipAngles = tf.euler_from_quaternion(ChestQuat, 'syzx') 
            
            HipPitch = HipAngles[2]
            HipRoll = HipAngles[1] 
                
            HipPitchList.append(HipPitch) 
            HipRollList.append(HipRoll)
            
            # Head angles.
            if (Joints[PyKinectV2.JointType_Neck].TrackingState == 2) and (Joints[PyKinectV2.JointType_Head].TrackingState == 2):
                NeckPos = Joints[PyKinectV2.JointType_Neck].Position
                HeadPos = Joints[PyKinectV2.JointType_Head].Position
                Diference = np.array([(HeadPos.x - NeckPos.x), (HeadPos.y - NeckPos.y), (HeadPos.z - NeckPos.z)])
                
                HeadPitch = np.arctan2(-Diference[2], Diference[1])
                HeadYaw = 0
                
            else:
                HeadPitch = None
                HeadYaw = None
                
            HeadPitchList.append(HeadPitch)
            HeadYawList.append(HeadYaw)
    
            # Right shoulder angles.
            if (Joints[PyKinectV2.JointType_ShoulderRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_ElbowRight].TrackingState == 2):
                ElbowRQuat = self.Quaternion(Orientations, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_SpineShoulder) 
                ShoulderRAngles = tf.euler_from_quaternion(ElbowRQuat, 'syzx')
                
                RShoulderPitch = add_angles(-np.pi/2, ShoulderRAngles[2])
                RShoulderRoll = -ShoulderRAngles[1]
            
                # Right elbow angles.
                if Joints[PyKinectV2.JointType_WristRight].TrackingState == 2:
                    WristRQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ElbowRight)   
                    ElbowRAngles = tf.euler_from_quaternion(WristRQuat, 'syzx') 
                    
                    RElbowYaw = -add_angles(np.pi, -ShoulderRAngles[0])
                    RElbowRoll = ElbowRAngles[1]
                    
                    # Right wrist angle.
                    if ENABLE_WRIST:
                        RWristYaw = -add_angles(np.pi/2, -ElbowRAngles[0]) 
                    
                    else:
                        RWristYaw = None
                  
                    # Right hand state.
                    if (Joints[PyKinectV2.JointType_HandTipRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_ThumbRight].TrackingState == 2):
                        if Body.hand_right_confidence == 1:
                            HandR = Body.hand_right_state
                        
                        else:
                            HandR = None
                    
                    else:
                        HandR = None

                else:
                    RElbowYaw = None
                    RElbowRoll = None

                    RWristYaw = None
                    
                    HandR = None
            
            else:
                RShoulderPitch = None
                RShoulderRoll = None
                
                RElbowYaw = None
                RElbowRoll = None
                
                RWristYaw = None
                
                HandR = None
                
            RShoulderPitchList.append(RShoulderPitch)
            RShoulderRollList.append(RShoulderRoll)
        
            RElbowRollList.append(RElbowRoll)
            RElbowYawList.append(RElbowYaw)

            RWristYawList.append(RWristYaw)
            
            RHandList.append(HandR)

            # Left shoulder angles.
            if (Joints[PyKinectV2.JointType_ShoulderLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_ElbowLeft].TrackingState == 2):
                ElbowLQuat = self.Quaternion(Orientations, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_SpineShoulder) 
                ShoulderLAngles = tf.euler_from_quaternion(ElbowLQuat, 'syzx') 
                
                LShoulderPitch = add_angles(-np.pi/2, ShoulderLAngles[2])
                LShoulderRoll = -ShoulderLAngles[1]

                # Left elbow angles.
                if Joints[PyKinectV2.JointType_WristLeft].TrackingState == 2:
                    WristLQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ElbowLeft)  
                    ElbowLAngles = tf.euler_from_quaternion(WristLQuat, 'syzx') 
                    
                    LElbowYaw =  -add_angles(np.pi, -ShoulderLAngles[0])
                    LElbowRoll = ElbowLAngles[1]
                    
                    # Left wrist angle.
                    if ENABLE_WRIST:
                        LWristYaw = -add_angles(np.pi/2, -ElbowLAngles[0])
                    
                    else:
                        LWristYaw = None

                    # Left hand state.
                    if (Joints[PyKinectV2.JointType_HandTipLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_ThumbLeft].TrackingState == 2):
                        if Body.hand_left_confidence == 1:
                            HandL = Body.hand_left_state
                        
                        else:
                            HandL = None
                    
                    else:
                        HandL = None
                            
                else:
                    LElbowYaw = None
                    LElbowRoll = None
                                            
                    
                    LWristYaw = None
                                                
                    HandL = None
                                    
            else:
                LShoulderPitch = None
                LShoulderRoll = None
                
                LElbowYaw = None
                LElbowRoll = None
                
                LWristYaw = None
                
                HandL = None
        
            LShoulderPitchList.append(LShoulderPitch)
            LShoulderRollList.append(LShoulderRoll)
        
            LElbowRollList.append(LElbowRoll)
            LElbowYawList.append(LElbowYaw)
            
            LWristYawList.append(LWristYaw)
            
            LHandList.append(HandL)

    def DrawBodyBones(self, Joints, JointsPoints, Color, Joint0, Joint1):
        '''
        Function that draw the skeleton in a frame.
        '''
        Joint0State = Joints[Joint0].TrackingState;
        Joint1State = Joints[Joint1].TrackingState;

        # If is not correct the tracking:
        if (Joint0State == PyKinectV2.TrackingState_NotTracked) or (Joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        if (Joint0State == PyKinectV2.TrackingState_Inferred) and (Joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # If the skeleton tracking is correct:
        Start = (JointsPoints[Joint0].x, JointsPoints[Joint0].y)                # Start point coordinates.
        End = (JointsPoints[Joint1].x, JointsPoints[Joint1].y)                  # Final point coordinates.

        # Draw a line with start and final points.
        try:
            if (Joint0State == PyKinectV2.TrackingState_Inferred) or (Joint1State == PyKinectV2.TrackingState_Inferred):
                pygame.draw.line(self.FrameSurface, Color[1], Start, End, 8)
            
            else:
                pygame.draw.line(self.FrameSurface, Color[0], Start, End, 8)
                
        except: 
            pass

    def Quaternion(self, Orientations, Joint, ParentJoint = None):
        '''
        Function that computes the quaternion matrix using the coordinates of
        each joint get with the Kinect Sensor.
        '''        
        Quat = Orientations[Joint].Orientation 
        QArray = np.array([Quat.w, Quat.x, Quat.y, Quat.z]) 
         
        # Quat matrix with two joints.
        if ParentJoint is not None:
            QuatParent = Orientations[ParentJoint].Orientation 
            QuatArrayParent = np.array([QuatParent.w, QuatParent.x, QuatParent.y, QuatParent.z]) 
            QuatRelativ = tf.quaternion_multiply(tf.quaternion_inverse(QuatArrayParent), QArray) # Compute the relative quat.
            
            return QuatRelativ
         
        else:           
            return QArray

    def SendToPepper(self, JointName, Angle):
        '''
        Function that do that Pepper put its joints in the specific orientation.
        '''
        self.Motion.setAngles(JointName, Angle, JointsParams[JointName]['fractionMaxSpeed'])

    def SendToPepperHands(self, Hand, State):
        '''
        Function that do that Pepper close and open its hands.
        '''
        if State == 2:
            self.Motion.setAngles(Hand, 1, JointsParams[Hand]['fractionMaxSpeed'])
        elif State == 3:
            self.Motion.setAngles(Hand, 0, JointsParams[Hand]['fractionMaxSpeed'])
        else:
            self.Motion.setAngles(Hand, 0.5, JointsParams[Hand]['fractionMaxSpeed']) 

    def InitPosition(self):
        '''
        Function that send Pepper's joints to "init" position.
        '''
        self.Posture.goToPosture('StandInit', 1.0)

    def GetToPepper(self,JointName):
        '''
        Function that read the Pepper's joints sensors.
        '''
        AngleRad = self.Motion.getAngles(JointName, True)
        AngleRad = float(AngleRad[0])
        return AngleRad

    '''
    The current functions verify and limit the range of motion of each joint 
    to the range allowed by Pepper.
    ------------------------------------------------------------------------
    The options with "LSup" and "LInf", have been calcualted through linear 
    interpolation.
    ------------------------------------------------------------------------
    Go further: see technical information of Pepper.
    ------------------------------------------------------------------------
    '''
    def VerifyHipAngles(self, JointName, Angle):
        if JointName == "HipPitch":
            if -1.0385 <= Angle <= 1.0385: 
                self.SendToPepper(JointName, Angle)
            elif -1.0385 > Angle: 
                self.SendToPepper(JointName, -1.0385)
            elif 1.0385 < Angle: 
                self.SendToPepper(JointName, 1.0385)
         
        if JointName == "HipRoll":
            if -0.5149 <= Angle <= 0.5149: 
                self.SendToPepper(JointName, Angle)
            elif -0.5149 > Angle: 
                self.SendToPepper(JointName, -0.5149)
            elif 0.5149 < Angle: 
                self.SendToPepper(JointName, 0.5149)
    
    def VerifyHeadYawAngles(self, Angle):
        if -2.0857 <= Angle <= 2.0857: 
            self.SendToPepper('HeadYaw', Angle)
        elif -2.0857 > Angle: 
            self.SendToPepper('HeadYaw', -2.0857)
        elif 2.0857 < Angle: 
            self.SendToPepper('HeadYaw', 2.0857)
    
    def VerifyHeadPitchAngles(self, Pitch, Yaw = 0):
        if (-119.5*almath.TO_RAD) <= Yaw < (-91.4*almath.TO_RAD): 
            if (-35.0*almath.TO_RAD) <= Pitch <= (13.5*almath.TO_RAD): 
                self.SendToPepper('HeadPitch', Pitch)
            elif (-35.0*almath.TO_RAD) > Pitch: 
                self.SendToPepper('HeadPitch', -35.0*almath.TO_RAD)
            elif (13.5*almath.TO_RAD) < Pitch: 
                self.SendToPepper('HeadPitch', 13.5*almath.TO_RAD)
        
        if (-91.4*almath.TO_RAD) <= Yaw < (-61.6*almath.TO_RAD):
              
            LInf = (-0.0034*Yaw) - 0.618
            LSup = (0.2483*Yaw) + 0.6318
              
            if LInf <= Pitch <= LSup: 
                self.SendToPepper('HeadPitch', Pitch)
            elif LInf > Pitch: 
                self.SendToPepper('HeadPitch', LInf)
            elif LSup < Pitch: 
                self.SendToPepper('HeadPitch', LSup)
              
        if (-61.6*almath.TO_RAD) <= Yaw < (-33.33*almath.TO_RAD):
            LInf = (-0.1875*Yaw) - 0.8159
            LSup = (0.5518*Yaw) + 0.958   
  
            if LInf <= Pitch <= LSup: 
                self.SendToPepper('HeadPitch', Pitch)
            elif LInf > Pitch: 
                self.SendToPepper('HeadPitch', LInf)
            elif LSup < Pitch: 
                self.SendToPepper('HeadPitch', LSup)
          
        if (-33.33*almath.TO_RAD) <= Yaw < (33.33*almath.TO_RAD):
            if (-40.5*almath.TO_RAD) <= Pitch <= (36.5*almath.TO_RAD):
                self.SendToPepper('HeadPitch', Pitch)
            elif (-40.5*almath.TO_RAD) > Pitch: 
                self.SendToPepper('HeadPitch', -40.5*almath.TO_RAD)
            elif (36.5*almath.TO_RAD) < Pitch: 
                self.SendToPepper('HeadPitch', 36.5*almath.TO_RAD)
  
        if (33.33*almath.TO_RAD) <= Yaw < (61.6*almath.TO_RAD):
            LInf = (0.1875*Yaw) - 0.8159
            LSup = (-0.5518*Yaw) + 0.958   
  
            if LInf <= Pitch <= LSup: 
                self.SendToPepper('HeadPitch', Pitch)
            elif LInf > Pitch: 
                self.SendToPepper('HeadPitch', LInf)
            elif LSup < Pitch: 
                self.SendToPepper('HeadPitch', LSup)
  
        if (61.6*almath.TO_RAD) <= Yaw < (91.4*almath.TO_RAD):
            LInf = (0.0034*Yaw) - 0.618
            LSup = (-0.2483*Yaw) + 0.6318
              
            if LInf <= Pitch <= LSup: 
                self.SendToPepper('HeadPitch', Pitch)
            elif LInf > Pitch:
                self.SendToPepper('HeadPitch', LInf)
            elif LSup < Pitch: 
                self.SendToPepper('HeadPitch', LSup)     
  
        if (91.4*almath.TO_RAD) <= Yaw <= (119.5*almath.TO_RAD): 
            if (-35.0*almath.TO_RAD) <= Pitch <= (13.5*almath.TO_RAD):
                self.SendToPepper('HeadPitch', Pitch)
            elif (-35.0*almath.TO_RAD) > Pitch: 
                self.SendToPepper('HeadPitch', -35.0*almath.TO_RAD)
            elif (13.5*almath.TO_RAD) < Pitch: 
                self.SendToPepper('HeadPitch', 13.5*almath.TO_RAD)     
            
    def VerifyShouldersAndWristsAngles(self, JointName, Angle):
        if (JointName == "LShoulderPitch"):
            if -2.0857 <= Angle <= 2.0857: 
                self.SendToPepper(JointName, Angle)
            elif -2.0857 > Angle: 
                self.SendToPepper(JointName, -2.0857)
            elif 2.0857 < Angle:
                self.SendToPepper(JointName, 2.0857)
  
        if (JointName == "RShoulderPitch"):
            if -2.0857 <= Angle <= 2.0857: 
                self.SendToPepper(JointName, Angle)
            elif -2.0857 > Angle: 
                self.SendToPepper(JointName, -2.0857)
            elif 2.0857 < Angle: 
                self.SendToPepper(JointName, 2.0857)
  
        if (JointName == "LShoulderRoll"):
            if 0.0087 <= Angle <= 1.5620: 
                self.SendToPepper(JointName, Angle)
            elif 0.0087 > Angle: 
                self.SendToPepper(JointName, 0.0087)
            elif 1.5620 < Angle: 
                self.SendToPepper(JointName, 1.5620)
  
        if (JointName == "RShoulderRoll"):
            if -1.5620 <= Angle <= -0.0087: 
                self.SendToPepper(JointName, Angle)
            elif -1.5620 > Angle:
                self.SendToPepper(JointName, -1.5620)
            elif -0.0087 < Angle: 
                self.SendToPepper(JointName, -0.0087)
  
        if (JointName == "LWristYaw"):
            if -1.8239 <= Angle <= 1.8239: 
                self.SendToPepper(JointName, Angle)
            elif -1.8239 > Angle: 
                self.SendToPepper(JointName, -1.8239)
            elif 1.8239 < Angle: 
                self.SendToPepper(JointName, 1.8239)
  
        if (JointName == "RWristYaw"):
            if -1.8239 <= Angle <= 1.8239: 
                self.SendToPepper(JointName, Angle)
            elif -1.8239 > Angle:
                self.SendToPepper(JointName, -1.8239)
            elif 1.8239 < Angle: 
                self.SendToPepper(JointName, 1.8239)
    
    def VerifyLElbowYawAngles(self, Angle):
        if -2.0857 <= Angle <= 2.0857: 
            self.SendToPepper('LElbowYaw', Angle)
        elif -2.0857 > Angle: 
            self.SendToPepper('LElbowYaw', -2.0857)
        elif 2.0857 < Angle: 
            self.SendToPepper('LElbowYaw', 2.0857) 

    def VerifyLElbowRollAngles(self, Roll, Yaw):
        if (-119.5*almath.TO_RAD) <= Yaw < (-60.0*almath.TO_RAD):
            if (-78.0*almath.TO_RAD) <= Roll <= (-0.5*almath.TO_RAD): 
                self.SendToPepper('LElbowRoll', Roll)
            elif(-78.0*almath.TO_RAD) > Roll: 
                self.SendToPepper('LElbowRoll', -78.0*almath.TO_RAD)
            elif (-0.5*almath.TO_RAD) < Roll: 
                self.SendToPepper('LElbowRoll', -0.5*almath.TO_RAD)
  
        if (-60.0*almath.TO_RAD) <= Yaw < (0.0*almath.TO_RAD):
            LInf = (-0.1917+Yaw) - 1.5621
            if LInf <= Roll <= (-0.5*almath.TO_RAD): 
                self.SendToPepper('LElbowRoll', Roll)
            elif LInf > Roll: 
                self.SendToPepper('LElbowRoll', LInf)
            elif (-0.5*almath.TO_RAD) < Roll: 
                self.SendToPepper('LElbowRoll', -0.5*almath.TO_RAD)
  
        if (0.0*almath.TO_RAD) <= Yaw < (99.5*almath.TO_RAD):
            if (-89.5*almath.TO_RAD) <= Roll <= (-0.5*almath.TO_RAD): 
                self.SendToPepper('LElbowRoll', Roll)
            elif (-89.5*almath.TO_RAD) > Roll: 
                self.SendToPepper('LElbowRoll', -89.5*almath.TO_RAD)
            elif (-0.5*almath.TO_RAD) < Roll: 
                self.SendToPepper('LElbowRoll', -0.5*almath.TO_RAD)
              
        if (99.5*almath.TO_RAD) <= Yaw <= (119.5*almath.TO_RAD):
            LInf = (0.325*Yaw) - 2.1265
            if  LInf <= Roll <= (-0.5*almath.TO_RAD): 
                self.SendToPepper('LElbowRoll', Roll)
            elif LInf > Roll: 
                self.SendToPepper('LElbowRoll', LInf)
            elif (-0.5*almath.TO_RAD) < Roll: 
                self.SendToPepper('LElbowRoll', -0.5*almath.TO_RAD)
  
    def VerifyRElbowYawAngles(self, Angle):
        if -2.0857 <= Angle <= 2.0857: 
            self.SendToPepper('RElbowYaw', Angle)
        elif -2.0857 > Angle: 
            self.SendToPepper('RElbowYaw', -2.0857)
        elif 2.0857 < Angle: 
            self.SendToPepper('RElbowYaw', 2.0857) 
    
    def VerifyRElbowRollAngles(self, Roll, Yaw):
        if (-119.5*almath.TO_RAD) <= Yaw < (-99.5*almath.TO_RAD):
            LSup = (0.325*Yaw) + 2.1265
  
            if (0.5*almath.TO_RAD) <= Roll <= LSup: 
                self.SendToPepper('RElbowRoll', Roll)
            elif(0.5*almath.TO_RAD) > Roll: 
                self.SendToPepper('RElbowRoll', 0.5*almath.TO_RAD)
            elif LSup < Roll: 
                self.SendToPepper('RElbowRoll', LSup)
          
        if (-99.5*almath.TO_RAD) <= Yaw < (0.0*almath.TO_RAD):
            if (0.5*almath.TO_RAD) <= Roll <= (89.5*almath.TO_RAD): 
                self.SendToPepper('RElbowRoll', Roll)
            elif (0.5*almath.TO_RAD) > Roll: 
                self.SendToPepper('RElbowRoll', 0.5*almath.TO_RAD)
            elif (89.5*almath.TO_RAD) < Roll: 
                self.SendToPepper('RElbowRoll', 89.5*almath.TO_RAD)
          
        if (0.0*almath.TO_RAD) <= Yaw < (60.0*almath.TO_RAD):
            LSup = (-0.1917*Yaw) + 1.5621
  
            if (0.5*almath.TO_RAD) <= Roll <= LSup: 
                self.SendToPepper('RElbowRoll', Roll)
            elif(0.5*almath.TO_RAD) > Roll: 
                self.SendToPepper('RElbowRoll', 0.5*almath.TO_RAD)
            elif LSup < Roll: 
                self.SendToPepper('RElbowRoll', LSup)
          
        if (60.0*almath.TO_RAD) <= Yaw < (119.5*almath.TO_RAD):
            if (0.5*almath.TO_RAD) <= Roll <= (78.0*almath.TO_RAD): 
                self.SendToPepper('RElbowRoll', Roll)
            elif(0.5*almath.TO_RAD) > Roll: 
                self.SendToPepper('RElbowRoll', 0.5*almath.TO_RAD)
            elif (78.0*almath.TO_RAD) < Roll:
                self.SendToPepper('RElbowRoll', 78.0*almath.TO_RAD)   

    def EndProgram(self):
        '''
        Function that end the tracking and close the program, before send Pepper
        to initial position.
        '''
        try:
            self.Btn_Recording.setText("REC")                                   # Change the text in the GUI Button.
            
            self.InitPosition()                                                 # Send Robot Pepper to init position.
            
            self.Kinect.close()                                                 # Stop Kinect Sensor.
            pygame.quit()
            
            self.VideoFile.release()                                            # End .avi file creation.
            
            sys.exit()                                                          # Close the GUI.  
        except:
            sys.exit()                                                          # Close the GUI.  
            pass
              
if __name__ == '__main__':
        
    App = QtGui.QApplication(sys.argv)
    GUI = GUI()
    try:
        sys.exit(GUI.InitPosition())
    except:
        sys.exit(App.exec_())

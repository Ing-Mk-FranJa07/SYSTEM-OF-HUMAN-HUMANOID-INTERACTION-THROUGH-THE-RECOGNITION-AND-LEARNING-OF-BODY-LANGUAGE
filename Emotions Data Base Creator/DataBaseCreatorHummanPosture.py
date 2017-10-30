'''
Created on 17/07/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This script allow create a data base of human postures associated with the mood
and emotions. 

The data type is eulerian angles gotten using the Kinect Sensor and computing 
the skeleton tracking, then are created .avi and .csv files that have the same
name defined by the user.

The Script works with a GUI developed to save new postures and replay the 
postures saved previously, also the GUI show the angles value of each joint.
'''
#coding: iso-8859-1

# The wrappers necessaries are imported.
from PyQt4 import uic, QtCore, QtGui                                            # Wrappers that allow used the GUI created.
from pykinect2 import PyKinectV2, PyKinectRuntime                               # Wrapper used to used the Kinect Sensor.
import pygame
import ctypes
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
WaistRollList = list(); WaistPitchList = list()                                 # Lists for waist angles.
RHipRollList = list(); RHipPitchList = list()                                   # Lists for right hip angles.
RKneeYawList = list(); RKneeRollList = list()                                   # Lists for right knee angles.
LHipRollList = list(); LHipPitchList = list()                                   # Lists for left hip angles.
LKneeYawList = list(); LKneeRollList = list()                                   # Lists for left knee angles.

Emotion = list()                                                                # List used to save the emotion that represent each posture.

# Is created a list with the header of each column used to create a .csv file.
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

ENABLE_WRIST = True                                                             # Is activated the tracking of the wrist.

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
    
class HumanPostureAndEmotion(QtGui.QWidget):
    '''
    Main class that allow the operation of the system using the GUI created, to
    show the image gotten using the Kinect Sensor, and compute the skeleton 
    tracking to get the body posture.
    '''
    def __init__(self):
        '''
        Function that initialize the GUI parameters and connect them with the
        corresponding functions.
        '''
        super(HumanPostureAndEmotion, self).__init__()
        
        # Is loaded the file that get the GUI.   
        self.MyGUI = uic.loadUi('D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBaseCreatorHumanPostureGUI.ui', self)

        # Are associated the different options that have the GUI with its corresponding functions.
        self.connect(self.Btn_Kinect, QtCore.SIGNAL("clicked()"), self.InitKinect)
        self.connect(self.Btn_REC, QtCore.SIGNAL("clicked()"), self.Recording)
        self.connect(self.Btn_Save, QtCore.SIGNAL("clicked()"), self.SavePosture)
        self.connect(self.Btn_Load, QtCore.SIGNAL("clicked()"), self.LoadPosture)
        self.connect(self.Btn_End, QtCore.SIGNAL("clicked()"), self.EndProgram)
        self.connect(self.Btn_Select_Emotion, QtCore.SIGNAL("clicked()"), self.SelectEmotion)
        self.connect(self.Check_Video, QtCore.SIGNAL("clicked()"), self.ActivateSaveVideo)
        self.connect(self.Check_Timer, QtCore.SIGNAL("clicked()"), self.ActivateRecordingTimer)            
        
        # Are created timers to have interruptions that allow execute functions.
        self.Timer = QtCore.QTimer(self.MyGUI)                                  # Timer that control the image tracking.
        self.Timer.timeout.connect(self.ShowFrame)                              # The timer active the Tracking function.
        
        self.Timer1 = QtCore.QTimer(self.MyGUI)                                 # Timer that control the playing of .avi file.
        self.Timer1.timeout.connect(self.ShowVideo)                             # The timer active the playing of .avi file.
    
        # Is created a pixel map to put in the GUI the image get with the Kinect Sensor.
        self.pixmap = QtGui.QPixmap()
        
        self.StartEnd = 1                                                       # Flag to start or end the Kinect Sensor. 

        self.MyGUI.show()
    
    def InitKinect(self):
        '''
        Function that start or End the Kinect Sensor.
        '''
        # Connect Kinect.
        if self.StartEnd == 1:
            # Are configured the parameters to use the Kinect Sensor.
            pygame.init()
            self.Clock = pygame.time.Clock()
            self.Kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
            self.FrameSurface = pygame.Surface((self.Kinect.color_frame_desc.Width, self.Kinect.color_frame_desc.Height), 0, 32)
    
            self.Bodies = None                                                  # Variable used to saved the bodies detected.
            
            self.Box_Emotion_Selector.setEnabled(True)                          # Enabled the emotions selection.
            self.Text_Save.setText("Digit File Name")                           # Indicate to the user that put the .avi and .cvs files names.
            self.Btn_Kinect.setText("Desconect Kinect")                         # Change the text in the GUI button.

            self.CreateVideoFile = 0                                            # Change flag's value to end .avi file creation.
            self.StartEnd = 0                                                   # Change flag's value to end the Kinect Sensor.                                  
            self.StartRec = 0                                                   # Change flag's value to end the recording.
            self.UseTimer = 1                                                   # Change flag's value that change the recording modes.
        
        # Disconnect Kinect.        
        else:
            try:
                self.Box_Emotion_Selector.setEnabled(False)                     # Disabled the emotions selection.
        
                self.Text_Save.setEnabled(False)                                # Disabled the text box to put the .avi and .csv files names.  
                
                self.Text_Save.setText("Digit File Name")                       # Indicate to the user that put the .avi and .cvs files names.
                
                self.Btn_Kinect.setText("Conect Kinect")                        # Change the text in the GUI button.
                
                self.Btn_REC.setText("REC")                                     # Change the text in the GUI button.
                
                self.StartEnd = 1                                               # Change flag's value to start the Kinect Sensor.
                
                self.Kinect.close()                                             # End the Kinect Sensor.
                
                pygame.quit()

                self.VideoFile.release()                                        # End the creation of .avi file.    
            except:
                pass
        
    def Recording(self):
        '''
        Function that allow the recording of the boy posture in a limited time.
        '''
        # Start recording.
        if self.StartRec == 0:
            self.Btn_REC.setText("STOP")                                        # Change the text in the GUI button.
            self.Btn_Save.setEnabled(False)                                     # Disabled the option that allow save .avi and .csv files.
            self.Text_Save.setEnabled(False)                                    # Disabled the option to put the name of the .avi and .csv files.    
            self.Text_Load.setEnabled(False)                                    # Disabled the option to that allow load .avi and .csv files.
            self.Btn_Load.setEnabled(False)                                     # Disabled the option to put the name to select .avi and .csv files saved.
            self.frame_2.setEnabled(True)                                       # Enabled the joint angles information.
            
            # Is created the .avi file.
            if self.CreateVideoFile == 1:
                self.CreateAviFile()
            
            # Are imported like global the lists used to save the information.  
            global HeadYawList, HeadRollList, HeadPitchList
             
            global RShoulderRollList, RShoulderPitchList
            global RElbowYawList, RElbowRollList
            global RWristYawList
            global RHandList
             
            global LShoulderRollList, LShoulderPitchList
            global LElbowYawList, LElbowRollList
            global LWristYawList
            global LHandList
     
            global WaistRollList, WaistPitchList
     
            global RHipRollList, RHipPitchList 
            global RKneeYawList, RKneeRollList
     
            global LHipRollList, LHipPitchList 
            global LKneeYawList, LKneeRollList
             
            global Emotion
            
            # Are cleared the lists used to save the information.
            HeadYawList = []; HeadRollList = []; HeadPitchList = []
            
            RShoulderRollList = []; RShoulderPitchList = []
            RElbowYawList = []; RElbowRollList = []
            RWristYawList = []
            RHandList = []
            
            LShoulderRollList = []; LShoulderPitchList = []
            LElbowYawList = []; LElbowRollList = []
            LWristYawList = []
            LHandList = []
            
            WaistRollList = []; WaistPitchList = []
            
            RHipRollList = []; RHipPitchList = []
            RKneeYawList = []; RKneeRollList = []
                
            LHipRollList = []; LHipPitchList = []
            LKneeYawList = []; LKneeRollList = []
    
            Emotion = []
            
            time.sleep(2)                                                       # Delay used to leave the user prepare him.
            
            self.Segundos = 0                                                   # Counter that limit the recording.
            
            self.Timer.start(1)                                                 # Activate the interruption that get the skeleton tracking.
            
            self.StartRec = 1                                                   # Change the flag's value to end the recording.
        
        # Stop recording.
        else:
            
            if self.UseTimer == 0:
                self.Timer.stop()                                               # Deactivate the interruption that get the skeleton tracking.                                               
                 
                try:                                                            # End the creation of the .avi file.
                    self.VideoFile.release()
                except AttributeError:
                    pass
                
                self.Btn_REC.setText("REC")                                     # Indicate the user that can start the recording.
                self.Btn_Save.setEnabled(True)                                  # Enabled the button that allow save the .avi and .csv files.
                self.Text_Save.setEnabled(True)                                 # Enabled the option to put the name of the .avi and .csv files.
                self.Box_Emotion_Selector.setEnabled(True)                      # Enabled the emotions selection.
                self.frame_2.setEnabled(False)                                  # Disabled the joint angles information.
                self.Text_Load.setEnabled(True)                                 # Enabled the text box to select with the name a .csv file saved.
                self.Btn_Load.setEnabled(True)                                  # Enabled the option to load a a .avi and .csv file saved.
                
            self.StartRec = 0                                               # Change the flag's value to start the recording.
                
    def SavePosture(self):
        '''
        Function that verify the saved angles value and create the .csv file.
        '''
        # The current lines verify each angle searching for a "None", if it is 
        # found is changed for the previous correct angle value in each array; 
        # The waist angles are not verify because the logic used in the function:
        # Angles.
        for M in range(0,len(WaistRollList)):
            
            # Head.
            if HeadPitchList[M] == None and HeadYawList[M] == None  and HeadRollList[M] == None:
                Flag = True
                I = 1
                while Flag == True:
                    if HeadPitchList[M-I] == None and HeadYawList[M-I] == None and HeadRollList[M-I] == None: 
                        I += 1
                    else: 
                        Flag = False
                        
                        HeadYawList[M] = HeadYawList[M-I]
                        HeadPitchList[M] = HeadPitchList[M-I]
                        HeadRollList[M] = HeadRollList[M-I]
                
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

            # Right hip.
            if RHipRollList[M] == None and RHipPitchList[M] == None:
                Flag = True
                I = 1
                while Flag == True:
                    if RHipRollList[M-I] == None and RHipPitchList[M-I] == None:
                        I += 1
                    else: 
                        Flag = False
                        
                        RHipRollList[M] = RHipRollList[M-I]
                        RHipPitchList[M] = RHipPitchList[M-I]

            # Right knee.
            if RKneeRollList[M] == None and RKneeYawList[M] == None:
                Flag = True
                I = 1
                while Flag == True:
                    if RKneeRollList[M-I] == None and RKneeYawList[M-I] == None:
                        I += 1
                    else: 
                        Flag = False
                        
                        RKneeRollList[M] = RKneeRollList[M-I]
                        RKneeYawList[M] = RKneeYawList[M-I]

            # Left hip.
            if LHipRollList[M] == None and LHipPitchList[M] == None:
                Flag = True
                I = 1
                while Flag == True:
                    if LHipRollList[M-I] == None and LHipPitchList[M-I] == None:
                        I += 1
                    else: 
                        Flag = False
                        
                        LHipRollList[M] = LHipRollList[M-I]
                        LHipPitchList[M] = LHipPitchList[M-I]

            # Left knee.
            if LKneeRollList[M] == None and LKneeYawList[M] == None:
                Flag = True
                I = 1
                while Flag == True:
                    if LKneeRollList[M-I] == None and LKneeYawList[M-I] == None: 
                        I += 1
                    else: 
                        Flag = False
                        
                        LKneeRollList[M] = LKneeRollList[M-I]
                        LKneeYawList[M] = LKneeYawList[M-I]
    
        # Is saved the name of the .csv file.
        FileName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\ " + str(self.Text_Save.text()) + ".csv")
        
        # Is created the .csv file.
        File = open(FileName, 'w') 
         
        # Is saved the header in the .csv file.
        for M in range(0,len(Header)):
            File.write(str(Header[M]))
            if M < len(Header)-1:
                File.write(",")
        
        File.write("\n")
        
        # Are saved in the .csv file the joint angles.
        for M in range(0,len(WaistPitchList)):
            
            File.write(str(HeadYawList[M])); File.write(","); File.write(str(HeadRollList[M])); File.write(","); File.write(str(HeadPitchList[M])); File.write(",")
            
            File.write(str(RShoulderRollList[M])); File.write(","); File.write(str(RShoulderPitchList[M])); File.write(",")
            File.write(str(RElbowYawList[M])); File.write(","); File.write(str(RElbowRollList[M])); File.write(",")
            File.write(str(RWristYawList[M])); File.write(",") 
            File.write(str(RHandList[M])); File.write(",");
            
            File.write(str(LShoulderRollList[M])); File.write(","); File.write(str(LShoulderPitchList[M])); File.write(",")
            File.write(str(LElbowYawList[M]));  File.write(","); File.write(str(LElbowRollList[M])); File.write(",")
            File.write(str(LWristYawList[M])); File.write(",") 
            File.write(str(LHandList[M])); File.write(",") 

            File.write(str(WaistRollList[M])); File.write(","); File.write(str(WaistPitchList[M])); File.write(",") 
             
            File.write(str(RHipRollList[M])); File.write(","); File.write(str(RHipPitchList[M])); File.write(",")
            File.write(str(RKneeYawList[M])); File.write(","); File.write(str(RKneeRollList[M])); File.write(",")
            
            File.write(str(LHipRollList[M])); File.write(","); File.write(str(LHipPitchList[M])); File.write(",") 
            File.write(str(LKneeYawList[M])); File.write(","); File.write(str(LKneeRollList[M])); File.write(",")
             
            File.write(str(Emotion[M]))
            
            File.write("\n")
            
        File.close()                                                            # Close the .csv file.
        
        self.Text_Save.setText("")                                              # Cleared the text box to put a new name to .csv file.
        self.Btn_Save.setEnabled(False)                                         # Disabled the option to create a new .csv file.
    
    def LoadPosture(self):
        '''
        Function that allow load a .avi and .csv file to be played.
        '''
        # Is disconnected the Kinect Sensor.
        try:
            self.Kinect.close()
            pygame.quit()
        except AttributeError:
            pass     
        
        self.StartEnd = 1                                                       # Change the flag's value to start the skeleton tracking.
        
        self.Btn_Kinect.setText("Conect Kinect")                                # Change the text in the GUI button.
        self.Btn_REC.setEnabled(False)                                          # Disabled the recording option.
        
        # Are imported like global the lists used to save the information.         
        global HeadYawList, HeadRollList, HeadPitchList
         
        global RShoulderRollList, RShoulderPitchList
        global RElbowYawList, RElbowRollList
        global RWristYawList
        global RHandList
         
        global LShoulderRollList, LShoulderPitchList
        global LElbowYawList, LElbowRollList
        global LWristYawList
        global LHandList
 
        global WaistRollList, WaistPitchList
 
        global RHipRollList, RHipPitchList 
        global RKneeYawList, RKneeRollList
 
        global LHipRollList, LHipPitchList 
        global LKneeYawList, LKneeRollList
        
        # Are cleared the lists used to save the information.
        HeadYawList = []; HeadRollList = []; HeadPitchList = []
        
        RShoulderRollList = []; RShoulderPitchList = []
        RElbowYawList = []; RElbowRollList = []
        RWristYawList = []
        RHandList = []
        
        LShoulderRollList = []; LShoulderPitchList = []
        LElbowYawList = []; LElbowRollList = []
        LWristYawList = []
        LHandList = []
        
        WaistRollList = []; WaistPitchList = []
        
        RHipRollList = []; RHipPitchList = []
        RKneeYawList = []; RKneeRollList = []
            
        LHipRollList = []; LHipPitchList = []
        LKneeYawList = []; LKneeRollList = []
        
        # Are loaded the .avi and .csv files
        try: 
            # .csv file.
            FileName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\ " + str(self.Text_Load.text()) + ".csv")
            File = pd.read_csv(FileName, header = 0)

            # .avi file.
            VideoName = str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\ " + str(self.Text_Load.text()) + ".avi")
            self.Video = cv2.VideoCapture(VideoName)

            FileLoad = 1                                                        # Check the correct load.
            
        except:
            FileLoad = 0                                                        # Check the incorrect load.
            pass
        
        # If the .avi and .csv files are loaded correctly:
        # the information of each angle is saved in the corresponding list, and 
        # then is played the .avi file while are showed the angles in the GUI.
        if FileLoad == 1:
            self.Text_Load.setText("")                                          # Cleared the text box to put the name of the .avi and .csv files.
            
            # Head angles.
            HeadYawList = File['Head Yaw']; HeadRollList = File['Head Roll']; HeadPitchList = File['Head Pitch']

            # Right arm angles.
            RShoulderRollList = File['Right Shoulder Roll']; RShoulderPitchList = File['Right Shoulder Pitch']
            RElbowYawList = File['Right Elbow Yaw']; RElbowRollList = File['Right Elbow Roll']
            RWristYawList = File['Right Wrist Yaw']
            RHandList = File['Right Hand']

            # Left arm angles.
            LShoulderRollList = File['Left Shoulder Roll']; LShoulderPitchList = File['Left Shoulder Pitch']
            LElbowYawList = File['Left Elbow Yaw']; LElbowRollList = File['Left Elbow Roll']
            LWristYawList = File['Left Wrist Yaw']
            LHandList = File['Left Hand']
            
            # Waist angles.
            WaistRollList = File['Waist Roll'];  WaistPitchList = File['Waist Pitch']
            
            # Right leg angles.
            RHipRollList = File['Right Hip Roll']; RHipPitchList = File['Right Hip Pitch']
            RKneeYawList = File['Right Knee Yaw']; RKneeRollList = File['Right Knee Roll']
 
            # Left leg angles.
            LHipRollList = File['Left Hip Roll']; LHipPitchList = File['Left Hip Pitch']
            LKneeYawList = File['Left Knee Yaw']; LKneeRollList = File['Left Knee Roll']
            
            # Time limit to played the .avi file.
            if len(HeadYawList) < 30:                               
                self.Limit = 27
            else:
                self.Limit = len(HeadPitchList) - 1

            self.Sec = 0                                                        # Counter used to play the .avi file.
            
            self.Timer1.start(1)                                                # Activate the interruption that show the .avi frames in the GUI.
        
        # Is the .avi and .csv files load is not correct.
        else:
            self.Text_Load.setText("Put a valid name")                          # The user is informed that the .avi and .csv files names are incorrect.
        
        self.Btn_Save.setEnabled(False)                                         # Disabled the option to create a new .csv file.
    
    def EndProgram(self):
        '''
        Function that end the tracking and close the program.
        '''
        try:
            self.Kinect.Stop_Kinect()                                           # Stop Kinect V2 Camera.
            pygame.quit()                                                       # End visualization of kinect's image.

            sys.exit()                                                          # Close the GUI.    
        except:
            sys.exit()                                                          # Close the GUI.  
              
    def SelectEmotion(self):
        '''
        Function that allow select the emotion that the user goes to represent.
        '''
        if self.RadioBtn_Happy.isChecked():
            self.EmotionAsocieted = "Happy"

        if self.RadioBtn_Sad.isChecked():
            self.EmotionAsocieted = "Sad"

        if self.RadioBtn_Angry.isChecked():
            self.EmotionAsocieted = "Angry"
            
        if self.RadioBtn_Surprised.isChecked():
            self.EmotionAsocieted = "Surprised"
            
        if self.RadioBtn_Reflexive.isChecked():
            self.EmotionAsocieted = "Reflexive"
            
        if self.RadioBtn_Insecure.isChecked():
            self.EmotionAsocieted = "Insecure"
            
        if self.RadioBtn_Trusted.isChecked():
            self.EmotionAsocieted = "Trusted"
            
        if self.RadioBtn_Normal.isChecked():
            self.EmotionAsocieted = "Normal"

        self.Check_Video.setEnabled(True)                                       # Enabled the option to create .avi file.
        self.Check_Timer.setEnabled(True)                                       # Enabled the recording mode.
        self.Text_Save.setEnabled(False)                                        # Disabled the option to put a name for .avi and .csv files.
        self.Box_Emotion_Selector.setEnabled(False)                             # Disabled the emotion selection.
        self.Btn_REC.setEnabled(True)                                           # Enabled the recording option.
            
    def ActivateSaveVideo(self):
        '''
        Function that allow the cration of .avi file.
        '''
        # If the .avi file creation option is selected:
        if self.Check_Video.isChecked():
            self.Text_Save.setEnabled(True)                                     # Enabled the text box to put the name of the .avi and .csv files.
            self.Btn_REC.setEnabled(True)                                       # Enabled the recordign option.
            
            self.CreateVideoFile = 1                                            # Change the flag's value to start the .avi file creation.
        
        # If the .avi file creation option is not selected.
        else:
            self.Btn_REC.setEnabled(False)                                      # Disabled the recording option.    
            self.Text_Save.setEnabled(False)                                    # Disabled the text box to put the name of the .avi and .csv files.          
            
            self.CreateVideoFile = 0                                            # Change the flag's value to end the .avi file creation.

    def ActivateRecordingTimer(self):
        '''
        Function that allow change the mode of the recording, with time limit or
        without time limit.
        '''
        # Activate the time limit:
        if self.Check_Timer.isChecked():
            self.UseTimer = 1                                                   # Change the flag's value to start the recording mode with time limit.    
        
        # Deactivate the time limit.
        else:
            self.UseTimer = 0                                                   # Change the flag's value to start the recording mode without time limit.
    
    def ShowVideo(self):
        '''
        Function that allow show in the GUI the frames loaded from the .avi file
        and also show the angles information loaded from the .csv file.
        '''
        Image = self.Video.read()[1]                                            # Load a frame from the .avi file.    
                    
        # Create a pixel map from the Kinect's image.
        Image = QtGui.QImage(Image, 1080, 630, 3240 , QtGui.QImage.Format_RGB888)
        self.pixmap.convertFromImage(Image.rgbSwapped())
        self.KinectFrame.setPixmap(self.pixmap)                                 # Show the Kinect's image in the GUI.
        
        # Show the angles information loaded from the .csv file in the GUI.
        self.Text_Hip_Pitch.setText(str("%.3f" %(WaistPitchList[self.Sec])))
        self.Text_Hip_Roll.setText(str("%.3f" %(WaistRollList[self.Sec])))
        
        self.Text_RHip_Pitch.setText(str("%.3f" %(RHipPitchList[self.Sec])))
        self.Text_RHip_Roll.setText(str("%.3f" %(RHipRollList[self.Sec])))
        
        self.Text_RKnee_Roll.setText(str("%.3f" %(RKneeRollList[self.Sec])))
        self.Text_RKnee_Yaw.setText(str("%.3f" %(RKneeYawList[self.Sec])))
        
        self.Text_LHip_Pitch.setText(str("%.3f" %(LHipPitchList[self.Sec])))
        self.Text_LHip_Roll.setText(str("%.3f" %(LHipRollList[self.Sec])))
        
        self.Text_LKnee_Roll.setText(str("%.3f" %(LKneeRollList[self.Sec])))
        self.Text_LKnee_Yaw.setText(str("%.3f" %(LKneeYawList[self.Sec])))
        
        self.Text_Head_Pitch.setText(str("%.3f" %(HeadPitchList[self.Sec])))
        self.Text_Head_Roll.setText(str("%.3f" %(HeadRollList[self.Sec])))
        self.Text_Head_Yaw.setText(str("%.3f" %(HeadYawList[self.Sec])))
        
        self.Text_RShoulder_Pitch.setText(str("%.3f" %(RShoulderPitchList[self.Sec])))
        self.Text_RShoulder_Roll.setText(str("%.3f" %(RShoulderRollList[self.Sec])))
        
        self.Text_RElbow_Yaw.setText(str("%.3f" %(RElbowYawList[self.Sec])))
        self.Text_RElbow_Roll.setText(str("%.3f" %(RElbowRollList[self.Sec])))
        
        self.Text_RWrist_Yaw.setText(str("%.3f" %(RWristYawList[self.Sec])))
        
        self.Text_RHand.setText(RHandList[self.Sec])
        
        self.Text_LShoulder_Pitch.setText(str("%.3f" %(LShoulderPitchList[self.Sec])))
        self.Text_LShoulder_Roll.setText(str("%.3f" %(LShoulderRollList[self.Sec])))
        
        self.Text_LElbow_Yaw.setText(str("%.3f" %(LElbowYawList[self.Sec])))
        self.Text_LElbow_Roll.setText(str("%.3f" %(LElbowRollList[self.Sec])))
        
        self.Text_LWrist_Yaw.setText(str("%.3f" %(LWristYawList[self.Sec])))
        
        self.Text_LHand.setText(LHandList[self.Sec])

        # End the playing of the .avi and .csv file.
        if self.Sec == self.Limit:
            self.Timer1.stop()
            self.Video.release()

        self.Sec += 1

    def ShowFrame(self):
        '''
        Main function that allow the skeleton tracking and get the body posture
        to save the joint angles and then create the .avi and .cvs files.
        '''
        # The current lines allow the recordign of the video in a limited time.
        if self.UseTimer == 1:
            self.Segundos += 1
             
            # Deactive the timer in the limit time.
            if self.Segundos == 29:
                self.Timer.stop()                                               # Deactivate the interruption that get the skeleton tracking.                                               
                 
                try:                                                            # End the creation of the .avi file.
                    self.VideoFile.release()
                except AttributeError:
                    pass
                
                self.Btn_REC.setText("REC")                                     # Indicate the user that can start the recording.
                self.Btn_Save.setEnabled(True)                                  # Enabled the button that allow save the .avi and .csv files.
                self.Text_Save.setEnabled(True)                                 # Enabled the option to put the name of the .avi and .csv files.
                self.Box_Emotion_Selector.setEnabled(True)                      # Enabled the emotions selection.
                self.frame_2.setEnabled(False)                                  # Disabled the joint angles information.
                self.Text_Load.setEnabled(True)                                 # Enabled the text box to select with the name a .csv file saved.
                self.Btn_Load.setEnabled(True)                                  # Enabled the option to load a a .avi and .csv file saved.
                
                self.StartRec = 0                                               # Change the flag's value to start the recording.
                 
                return

        # Is reshape the Kinect's image in the format that allow cv2.
        self.Image = pygame.surfarray.array3d(self.FrameSurface)                # Pygame's surface is converted in an array. 
        self.Image = np.rollaxis(self.Image, 0, 2)                              # Reform the structure of the array.
         
        # Is reformat and reshape the Image to be put in the GUI.
        self.Image = cv2.cvtColor(self.Image, cv2.COLOR_RGB2BGR)
        self.Image = cv2.resize(self.Image,(1080,630), interpolation = cv2.INTER_CUBIC)
        
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
        
        # Is the .avi file option is selected.
        if self.CreateVideoFile == 1:
            self.VideoFile.write(self.Image)                                    # Save a new frame in the .avi file.
        
        # Create a pixel map from the Kinect's image.
        self.Image = QtGui.QImage(self.Image, 1080, 630, 3240 , QtGui.QImage.Format_RGB888)
        self.pixmap.convertFromImage(self.Image.rgbSwapped())                   # Specify a RGB format.
        self.KinectFrame.setPixmap(self.pixmap)                                 # Show the Kinect's image in the GUI.
        
        self.Clock.tick(60)

    def CreateAviFile(self):
        '''
        Function that create .avi file with the format selected.
        '''
        # Is saved the name of the .avi file
        VideoFilePath =  str("D:\Tesis\Python\Liclipse\Tesis\TrackingKinect\DataBaseCreatorHumanPosture\DataBasePostures\ " + str(self.Text_Save.text()) + ".avi")
        
        # Is defined the format of the file.
        VideoFileType = cv2.VideoWriter_fourcc('M','J','P','G')                 
        
        # Is created the .avi file.
        self.VideoFile = cv2.VideoWriter(VideoFilePath, VideoFileType, 10.0, (1080,630))

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
        
        The angles are showed in the GUI.
        '''  
        # Hip angles.
        if (Joints[PyKinectV2.JointType_SpineShoulder].TrackingState == 2) and (Joints[PyKinectV2.JointType_SpineBase].TrackingState == 2):
            Emotion.append(self.EmotionAsocieted)                               # Save the name of the emotion that is represented.           
            
            ChestQuat = self.Quaternion(Orientations, PyKinectV2.JointType_SpineBase) 
            HipAngles = tf.euler_from_quaternion(ChestQuat, 'syzx') 
            
            WaistPitch = HipAngles[2]; self.Text_Hip_Pitch.setText(str("%.3f" %(WaistPitch)))
            WaistRoll = HipAngles[1]; self.Text_Hip_Roll.setText(str("%.3f" %(WaistRoll)))
                
            WaistPitchList.append(WaistPitch) 
            WaistRollList.append(WaistRoll)
                       
            # Right leg angles.
            if (Joints[PyKinectV2.JointType_HipRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_KneeRight].TrackingState == 2):
                LegQuat = self.Quaternion(Orientations, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_HipRight )
                LegAngles = tf.euler_from_quaternion(LegQuat, 'syzx')
                
                RHipPitch = LegAngles[2]; self.Text_RHip_Pitch.setText(str("%.3f" %(RHipPitch)))
                RHipRoll = LegAngles[1]; self.Text_RHip_Roll.setText(str("%.3f" %(RHipRoll)))
                 
                # Right knee angles.
                if (Joints[PyKinectV2.JointType_AnkleRight].TrackingState == 2):
                    KneeQuat = self.Quaternion(Orientations, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_KneeRight)
                    KneeAngles = tf.euler_from_quaternion(KneeQuat, 'syzx')
                    
                    RKneeRoll = KneeAngles[1]; self.Text_RKnee_Roll.setText(str("%.3f" %(RKneeRoll)))
                    RKneeYaw = KneeAngles[0]; self.Text_RKnee_Yaw.setText(str("%.3f" %(RKneeYaw)))
                
                else:
                    RKneeRoll = None
                    RKneeYaw = None
            
            else:
                RKneeRoll = None
                RKneeYaw = None

                RHipPitch = None
                RHipRoll = None          
            
            RKneeRollList.append(RKneeRoll)
            RKneeYawList.append(RKneeYaw)
            
            RHipPitchList.append(RHipPitch)
            RHipRollList.append(RHipRoll)

            # Left leg angles.
            if (Joints[PyKinectV2.JointType_HipLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_KneeLeft].TrackingState == 2):
                LegQuat = self.Quaternion(Orientations, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_HipLeft)
                LegAngles = tf.euler_from_quaternion(LegQuat, 'syzx')
                
                LHipPitch = LegAngles[2]; self.Text_LHip_Pitch.setText(str("%.3f" %(LHipPitch)))
                LHipRoll = LegAngles[1]; self.Text_LHip_Roll.setText(str("%.3f" %(LHipRoll)))
                
                # Left knee angles.
                if (Joints[PyKinectV2.JointType_AnkleLeft].TrackingState == 2):
                    KneeQuat = self.Quaternion(Orientations, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_KneeLeft)
                    KneeAngles = tf.euler_from_quaternion(KneeQuat, 'syzx')
                    
                    LKneeRoll = KneeAngles[1]; self.Text_LKnee_Roll.setText(str("%.3f" %(LKneeRoll)))
                    LKneeYaw = KneeAngles[0]; self.Text_LKnee_Yaw.setText(str("%.3f" %(LKneeYaw)))
                
                else:
                    LKneeRoll = None
                    LKneeYaw = None
            
            else:
                LKneeRoll = None
                LKneeYaw = None

                LHipPitch = None
                LHipRoll = None            
            
            LKneeRollList.append(LKneeRoll)
            LKneeYawList.append(LKneeYaw)
            
            LHipPitchList.append(LHipPitch)
            LHipRollList.append(LHipRoll)
            
            # Head angles.
            if (Joints[PyKinectV2.JointType_Neck].TrackingState == 2) and (Joints[PyKinectV2.JointType_Head].TrackingState == 2):
                NeckPos = Joints[PyKinectV2.JointType_Neck].Position
                HeadPos = Joints[PyKinectV2.JointType_Head].Position
                Diference = np.array([(HeadPos.x - NeckPos.x), (HeadPos.y - NeckPos.y), (HeadPos.z - NeckPos.z)])
                
                Pitch = np.arctan2(-Diference[2], Diference[1]); self.Text_Head_Pitch.setText(str("%.3f" %(Pitch)))
                Roll = 0; self.Text_Head_Roll.setText(str("%.3f" %(Roll)))
                Yaw = 0; self.Text_Head_Yaw.setText(str("%.3f" %(Yaw)))
                
            else:
                Pitch = None
                Roll = None
                Yaw = None
                
            HeadPitchList.append(Pitch)
            HeadYawList.append(Yaw)
            HeadRollList.append(Roll)
    
            # Right shoulder angles.
            if (Joints[PyKinectV2.JointType_ShoulderRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_ElbowRight].TrackingState == 2):
                ElbowRQuat = self.Quaternion(Orientations, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_SpineShoulder) 
                ShoulderRAngles = tf.euler_from_quaternion(ElbowRQuat, 'syzx')
                
                RShoulderPitch = add_angles(-np.pi/2, ShoulderRAngles[2]); self.Text_RShoulder_Pitch.setText(str("%.3f" %(RShoulderPitch)))
                RShoulderRoll = -ShoulderRAngles[1]; self.Text_RShoulder_Roll.setText(str("%.3f" %(RShoulderRoll)))
            
                # Right elbow angles.
                if Joints[PyKinectV2.JointType_WristRight].TrackingState == 2:
                    WristRQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ElbowRight)   
                    ElbowRAngles = tf.euler_from_quaternion(WristRQuat, 'syzx') 
                    
                    RElbowYaw = -add_angles(np.pi, -ShoulderRAngles[0]); self.Text_RElbow_Yaw.setText(str("%.3f" %(RElbowYaw)))
                    RElbowRoll = ElbowRAngles[1]; self.Text_RElbow_Roll.setText(str("%.3f" %(RElbowRoll)))
                    
                    # Right wrist angle.
                    if ENABLE_WRIST:
                        WristQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristRight)
                        WristAngles = tf.euler_from_quaternion(WristQuat, 'syzx')
                        
                        RWristYaw = WristAngles[0]; self.Text_RWrist_Yaw.setText(str("%.3f" %(RWristYaw)))
                    
                    else:
                        RWristYaw = None
                  
                    # Right hand state.
                    if (Joints[PyKinectV2.JointType_HandTipRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_ThumbRight].TrackingState == 2):
                            if Body.hand_right_state == 3:
                                HandR = 2                                       # Hand closed.
                            elif Body.hand_right_state == 2:
                                HandR = 1                                       # Hand opened.
                            else:
                                HandR = 3                                       # Unknown state.
                            
                            self.Text_RHand.setText(HandR)
                    
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
                
                LShoulderPitch = add_angles(-np.pi/2, ShoulderLAngles[2]); self.Text_LShoulder_Pitch.setText(str("%.3f" %(LShoulderPitch)))
                LShoulderRoll = -ShoulderLAngles[1]; self.Text_LShoulder_Roll.setText(str("%.3f" %(LShoulderRoll)))

                # Left elbow angles.
                if Joints[PyKinectV2.JointType_WristLeft].TrackingState == 2:
                    WristLQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ElbowLeft)  
                    ElbowLAngles = tf.euler_from_quaternion(WristLQuat, 'syzx') 
                    
                    LElbowYaw =  -add_angles(np.pi, -ShoulderLAngles[0]); self.Text_LElbow_Yaw.setText(str("%.3f" %(LElbowYaw)))
                    LElbowRoll = ElbowLAngles[1]; self.Text_LElbow_Roll.setText(str("%.3f" %(LElbowRoll)))
                    
                    # Left wrist angle.
                    if ENABLE_WRIST:
                        WristQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristLeft)
                        WristAngles = tf.euler_from_quaternion(WristQuat, 'syzx')
                        
                        LWristYaw = WristAngles[0]; self.Text_LWrist_Yaw.setText(str("%.3f" %(LWristYaw)))
                    
                    else:
                        LWristYaw = None

                    # Left hand state.
                    if (Joints[PyKinectV2.JointType_HandTipLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_ThumbLeft].TrackingState == 2):
                            if Body.hand_left_state == 3:
                                HandL = 2                                       # Hand closed.
                            elif Body.hand_left_state == 2:
                                HandL = 1                                       # Hand opened
                            else:
                                HandL = 3                                       # Unknown state.
                            
                            self.Text_LHand.setText(HandL)
                    
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
    
if __name__ == '__main__':
    
    App = QtGui.QApplication(sys.argv)
    GUI = HumanPostureAndEmotion()
    sys.exit(App.exec_()

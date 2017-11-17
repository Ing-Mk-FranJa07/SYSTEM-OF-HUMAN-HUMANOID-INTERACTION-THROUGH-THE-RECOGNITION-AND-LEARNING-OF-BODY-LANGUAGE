'''
Created on 10/08/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

Script that allow in real time recognize the mood of one person in classify it
in six categories: Happy, Sad, Angry, Surprised, Reflexive and Normal state.

The classification is done using the kinect sensor to get the information of the
body posture and then this information is compute by a neural network pre-trained.

The mood is represented by emojies in a GUI created to do the feedback to the user.

Also, the script use LSTM neural networks models trained previously to create a 
chatbox, using a system that transform the speech into a text and then generate 
an answer that is showed in the GUI.
'''
#coding: iso-8859-1

# The wrappers necessaries are imported. 
from PyQt4 import uic, QtCore, QtGui                                            # Wrappers that allow used the GUI created.
import Emojies_rc                                                               # Wrapper used to show the emojies.
from pykinect2 import PyKinectV2, PyKinectRuntime                               # Wrapper used to used the Kinect Sensor.
import pygame
import ctypes
import cv2
import numpy as np
from math import pi
import transformations as tf                                                    # Wrapper used to handle matrices.
import speech_recognition as sr                                                 # Wrapper used to recognize the speech and transform it into a text.
from keras.models import load_model                                             # Wrapper used to import the neural network model pre-trained.

# Wrappers used to implement the LSTM neural networks models that allow generate answers to an input text.
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

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

Rec = sr.Recognizer()                                                           # Object used to get access to the microphone.

# Is created a model to generate a chatbox.
PepperSay = ChatBot('Pepper Answers', 
                     logic_adapter = ["chatterbot.logic.MathematicalEvualation",
                                      "chatterbot.logic.TimeLogicAdapter",
                                      "chatterbot.logic.BestMatch"])

# Is trained the chatbox model.
PepperSay.set_trainer(ChatterBotCorpusTrainer)
PepperSay.train("chatterbot.corpus.Pepper_Speech")

# Is defined a vector that has the parameters to draw the skeleton.
SkeletonColors = [pygame.color.THECOLORS["green"],
                  pygame.color.THECOLORS["red"]]

# Is loaded the neural network model.
RNA = load_model('...\Model_RNA_Recognition_Of_Emotions')

# Is activated the tracking of the wrist.
ENABLE_WRIST = True

def Add_Angles(alpha, beta):
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

class EmotionsRecognition(QtGui.QWidget):
    '''
    Main class that allow the operation of the system using the GUI created, to
    show the image gotten using the Kinect Sensor, and compute the skeleton tracking
    information to classify the mood of the person in front the kinect.
    '''
    def __init__(self):
        '''
        Function that initialize the GUI parameters and connect them with the 
        corresponding functions.
        '''
        super(EmotionsRecognition, self).__init__()
                
        # Is loaded the file that get the GUI.    
        self.MyGUI = uic.loadUi('...\RecognitionOfEmotionsGUI.ui', self)
              
        # Are created timers to have interruptions that allow execute functions.
        self.Timer = QtCore.QTimer(self.MyGUI)                                  # Timer that control the image tracking.
        self.Timer.timeout.connect(self.ShowFrame)                              # The timer active the Tracking function.
        
        self.Timer1 = QtCore.QTimer(self.MyGUI)                                 # Timer that control the speech recognition.
        self.Timer1.timeout.connect(self.GetAudio)                              # The timer active GetAudio function.
        
        # Are associated the different options that have the GUI with its corresponding functions.
        self.connect(self.Btn_ConectKinect, QtCore.SIGNAL("clicked()"), self.InitKinect)
        self.connect(self.CheckBox_Speech, QtCore.SIGNAL("clicked()"), self.SpeechRecognition)
        
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
            
            self.Btn_ConectKinect.setText("Desconect Kinect")                   # Change the text in the GUI button.

            self.StartEnd = 0                                                   # Change flag's value to end the Kinect Sensor.
            
            self.HideEmojies()                                                  # Hide the representations of the emotions in the GUI.
            
            self.Timer.start(1)                                                 # Active the interruption that show the image get with the Kinect Sensor.
        
        # Disconnect Kinect.          
        else:
            try:                
                self.Btn_ConectKinect.setText("Conect Kinect")                  # Change the text in the GUI button.

                self.StartEnd = 1                                               # Change the value flag's value to start the Kinect Sensor.

                self.Timer.stop()                                               # Deactivate the interruption that show the image get with the Kinect Sensor.
                
                self.Kinect.close()                         
                
                pygame.quit()
                
                self.ShowEmojies()                                              # Show the representations of the emotions in the GUI.
            except:
                pass
    
    def ShowFrame(self):
        '''
        Main function that allow the tracking and get the body's joint angles to 
        determine the mood.
        '''
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
        
        # Create a pixel map from the Kinect's image.
        self.Image = QtGui.QImage(self.Image, 1080, 630, 3240 , QtGui.QImage.Format_RGB888)
        self.pixmap.convertFromImage(self.Image.rgbSwapped())                   # Specify a RGB format.
        self.KinectFrame.setPixmap(self.pixmap)                                 # Show the Kinect's image in the GUI.
        
        self.Clock.tick(60)

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
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight)        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight)

        # Left Leg
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft)
        self.DrawBodyBones(Joints, JointsPoints, Color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft)

    def Angles(self, Joints, Orientations, Body):
        '''
        Function that calculated the joint's angles of the skeleton get with the
        Kinect Sensor, getting the quaternion matrix and then calculating the
        eulerian angles.
        
        If the skeleton tracking is not correct, complete or partial, the 
        corresponding angles are define like "None".
        '''     
        # Hip angles.
        if (Joints[PyKinectV2.JointType_SpineShoulder].TrackingState == 2) and (Joints[PyKinectV2.JointType_SpineBase].TrackingState == 2):          
            ChestQuat = self.Quaternion(Orientations, PyKinectV2.JointType_SpineBase) 
            HipAngles = tf.euler_from_quaternion(ChestQuat, 'syzx')                  
            
            WaistPitch = HipAngles[2]
            WaistRoll = HipAngles[1]
                       
            # Right Leg angles.
            if (Joints[PyKinectV2.JointType_HipRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_KneeRight].TrackingState == 2):
                LegQuat = self.Quaternion(Orientations, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_HipRight )
                LegAngles = tf.euler_from_quaternion(LegQuat, 'syzx')
                
                RHipPitch = LegAngles[2]
                RHipRoll = LegAngles[1]
                 
                # Right Knee angles.
                if (Joints[PyKinectV2.JointType_AnkleRight].TrackingState == 2):
                    KneeQuat = self.Quaternion(Orientations, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_KneeRight)
                    KneeAngles = tf.euler_from_quaternion(KneeQuat, 'syzx')

                    RKneeRoll = KneeAngles[1]
                    RKneeYaw = KneeAngles[0]
                
                else:
                    RKneeRoll = None
                    RKneeYaw = None
            
            else:
                RKneeRoll = None
                RKneeYaw = None

                RHipPitch = None
                RHipRoll = None

            # Left Leg angles.
            if (Joints[PyKinectV2.JointType_HipLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_KneeLeft].TrackingState == 2):
                LegQuat = self.Quaternion(Orientations, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_HipLeft)
                LegAngles = tf.euler_from_quaternion(LegQuat, 'syzx')

                LHipPitch = LegAngles[2]
                LHipRoll = LegAngles[1]
                
                # Left Knee angles.
                if (Joints[PyKinectV2.JointType_AnkleLeft].TrackingState == 2):
                    KneeQuat = self.Quaternion(Orientations, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_KneeLeft)
                    KneeAngles = tf.euler_from_quaternion(KneeQuat, 'syzx')

                    LKneeRoll = KneeAngles[1]
                    LKneeYaw = KneeAngles[0]
                
                else:
                    LKneeRoll = None
                    LKneeYaw = None
            
            else:
                LKneeRoll = None
                LKneeYaw = None

                LHipPitch = None
                LHipRoll = None            
            
            # Head angles
            if (Joints[PyKinectV2.JointType_Neck].TrackingState == 2) and (Joints[PyKinectV2.JointType_Head].TrackingState == 2):
                NeckPos = Joints[PyKinectV2.JointType_Neck].Position
                HeadPos = Joints[PyKinectV2.JointType_Head].Position

                Diference = np.array([(HeadPos.x - NeckPos.x), (HeadPos.y - NeckPos.y), (HeadPos.z - NeckPos.z)]) # Pitch angle is calculates with head and Neck coordinates.
                
                HeadPitch = np.arctan2(-Diference[2], Diference[1])
            
            else:
                HeadPitch = None
    
            # Right Shoulder angles.
            if (Joints[PyKinectV2.JointType_ShoulderRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_ElbowRight].TrackingState == 2):
                ElbowRQuat = self.Quaternion(Orientations, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_SpineShoulder) 
                ShoulderRAngles = tf.euler_from_quaternion(ElbowRQuat, 'syzx')
                
                RShoulderPitch = Add_Angles(-np.pi/2, ShoulderRAngles[2])
                RShoulderRoll = -ShoulderRAngles[1]
            
                # Right Elbow angles.
                if Joints[PyKinectV2.JointType_WristRight].TrackingState == 2:
                    WristRQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ElbowRight)   
                    ElbowRAngles = tf.euler_from_quaternion(WristRQuat, 'syzx') 
                    
                    RElbowYaw = -Add_Angles(np.pi, -ShoulderRAngles[0])
                    RElbowRoll = ElbowRAngles[1]
                    
                    # Right Wrist angle.
                    if ENABLE_WRIST:
                        
                        WristQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristRight)
                        WristAngles = tf.euler_from_quaternion(WristQuat, 'syzx')
                        
                        RWristYaw = WristAngles[0]
                    
                    else:
                        RWristYaw = None
                  
                    # Right Hand state.
                    if (Joints[PyKinectV2.JointType_HandTipRight].TrackingState == 2) and (Joints[PyKinectV2.JointType_ThumbRight].TrackingState == 2):
                            if Body.hand_right_state == 3:
                                HandR = 2                                       # Hand closed.
                            elif Body.hand_right_state == 2:
                                HandR = 1                                       # Hand opened.
                            else:
                                HandR = 3                                       # Unknown state.
                            
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

            # Left Shoulder angles.
            if (Joints[PyKinectV2.JointType_ShoulderLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_ElbowLeft].TrackingState == 2):
                ElbowLQuat = self.Quaternion(Orientations, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_SpineShoulder) 
                ShoulderLAngles = tf.euler_from_quaternion(ElbowLQuat, 'syzx') 
                
                LShoulderPitch = Add_Angles(-np.pi/2, ShoulderLAngles[2])
                LShoulderRoll = -ShoulderLAngles[1]

                # Left Elbow angles.
                if Joints[PyKinectV2.JointType_WristLeft].TrackingState == 2:
                    WristLQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ElbowLeft)  
                    ElbowLAngles = tf.euler_from_quaternion(WristLQuat, 'syzx') 
                    
                    LElbowYaw =  -Add_Angles(np.pi, -ShoulderLAngles[0])
                    LElbowRoll = ElbowLAngles[1]
                    
                    # Left Wrist angles.
                    if ENABLE_WRIST:
                        WristQuat = self.Quaternion(Orientations, PyKinectV2.JointType_WristLeft)
                        WristAngles = tf.euler_from_quaternion(WristQuat, 'syzx')
                        
                        LWristYaw = WristAngles[0]
                    
                    else:
                        LWristYaw = None

                    # Left Hand state.
                    if (Joints[PyKinectV2.JointType_HandTipLeft].TrackingState == 2) and (Joints[PyKinectV2.JointType_ThumbLeft].TrackingState == 2):
                            if Body.hand_left_state == 3:
                                HandL = 2                                       # Hand closed.
                            elif Body.hand_left_state == 2:
                                HandL = 1                                       # Hand opened
                            else:
                                HandL = 3                                       # Unknown state.
                    
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
                
            # Are saved all the joint angles in a vector.
            self.RNA_Data_Input = np.transpose([[HeadPitch], 
                                                [RShoulderRoll], [RShoulderPitch], [RElbowYaw], [RElbowRoll], [RWristYaw], [HandR], 
                                                [LShoulderRoll], [LShoulderPitch], [LElbowYaw], [LElbowRoll], [LWristYaw], [HandL],
                                                [WaistRoll], [WaistPitch],
                                                [RHipRoll], [RHipPitch], [RKneeYaw], [RKneeRoll],
                                                [LHipRoll], [LHipPitch], [LKneeYaw], [LKneeRoll]]) 

            # If the skeleton tracking is correct, the mood is determinate.
            if not None in self.RNA_Data_Input:
                self.RecognitionOfEmotion()                                     # Function that recognize the mood.
            
            # If the skeleton tracking is not correct, the emojies are hidden.
            else:
                self.HideEmojies()

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
    
    def RecognitionOfEmotion(self):
        '''
        Function that allow recognize the emotions of one person, implementing a
        neural network
        '''
        Output = RNA.predict(self.RNA_Data_Input)                               # The mood is recognized.

        Output_RNA = [round(Output[0][I]) for I in range(Output.shape[1])]      # The neural network output is round out to binary values.
        
        self.ShowEmotion(Output_RNA)                                            # The emotion is showed in the GUI:
    
    def ShowEmotion(self, EmotionCode):
        '''
        Function that show the mood of the person in the GUI using emojies to
        represent the mood.
        '''        
        # Show Happy:
        if np.all(EmotionCode == np.array([0,0,0,0,0,1])): 
            self.HideEmojies()
            self.Frame_Happy.show(); self.Lbl_Happy.show()

        # Show Sad:
        elif np.all(EmotionCode == np.array([0,0,0,0,1,0])):
            self.HideEmojies()
            self.Frame_Sad.show(); self.Lbl_Sad.show()

        # Show Angry
        elif np.all(EmotionCode == np.array([0,0,0,1,0,0])):
            self.HideEmojies()
            self.Frame_Angry.show(); self.Lbl_Angry.show()
            
        # Show Surprised
        elif np.all(EmotionCode == np.array([0,0,1,0,0,0])):
            self.HideEmojies()
            self.Frame_Surprised.show(); self.Lbl_Surprised.show()
            
        # Show Reflexive
        elif np.all(EmotionCode == np.array([0,1,0,0,0,0])):
            self.HideEmojies()
            self.Frame_Reflexive.show(); self.Lbl_Reflexive.show()
            
        # Show Normal:
        elif np.all(EmotionCode == np.array([1,0,0,0,0,0])):
            self.HideEmojies()
            self.Frame_Normal.show(); self.Lbl_Normal.show()
            
        # Mood no determined:
        else:
            self.HideEmojies()    
        
    def SpeechRecognition(self):
        '''
        Function that allow activate the speech recognition.
        '''
        if self.CheckBox_Speech.isChecked():
            self.CheckBox_Speech.setText("Speech Recognition Activated")        # Inform that the speech recognition is activated.
                        
            # The skeleton tracking is terminated. 
            self.StartEnd = 0
            self.InitKinect()
            self.Btn_ConectKinect.setEnabled(False)                                 
            
            # Calibrate the microphone to filter environmental noise.
            with sr.Microphone() as source:
                self.Lbl_Listening.setText("Please wait. Calibrating microphone...")
                print("Please wait. Calibrating microphone...")
                Rec.adjust_for_ambient_noise(source, duration = 3)              

            self.Timer1.start(100)                                              # Activate the interruption that get the speech.
        
        else:
            self.Lbl_Listening.setText(" ")                                     # Clear the text box.
            self.Lbl_PrintAnswer.setText(" ")                                   # Clear the text box.
            self.CheckBox_Speech.setText("Speech Recognition deactivated")      # Inform that the speech recognition is deactivated.
            
            self.Timer1.stop()                                                  # Deactivate the interruption that get the speech.
        
            self.Btn_ConectKinect.setEnabled(True)                              # The option that activate the tracking is enabled.
 
    def GetAudio(self):
        '''
        Function that allow do the speech recognition getting the audio through 
        the pc's microphone and transforming it into a text, to send it at the LSTM 
        model use to predict the "answer". 
        '''
        self.Lbl_PrintAnswer.setText(" ")                                       # Clear the text box.

        # Activate the microphone.
        with sr.Microphone() as source:
            self.Lbl_Listening.setText("Listening ..."); print("Listening ...") # Show in the GUI a informative message.
            pass
            Audio = Rec.listen(source)                                          # Capture the audio data.
            pass      

        # If the audio data was catch, is generate an "answer".
        try:
            self.Lbl_Listening.setText("Listened !"); print("Listened !")       # Show in the GUI a informative message.
            
            Question = Rec.recognize_google(Audio)                              # Transform the audio data to text.
            Answer = PepperSay.get_response(Question)                           # Get the "answer", and the motion animation consistently with the Pepper's answer.
            
            self.Lbl_Listening.setText(Question)                                # Show in the GUI the speech recognized.
            self.Lbl_PrintAnswer.setText(str(Answer))                           # Show in the GUI the "answer".
            
            print("    Your speech: " + Question)
            print("    Pepper's answer: " + str(Answer))
            
        # If the audio data was not catch correctly, the user is informed.
        except sr.UnknownValueError:
            self.Lbl_Listening.setText("No Listened !"); print("No Listened !")             
            self.Lbl_PrintAnswer.setText(" ")            
        except sr.RequestError as e:
            self.Lbl_Listening.setText('Audio error; {0}'.format(e)); print('Audio error; {0}'.format(e))
            self.Lbl_PrintAnswer.setText(" ")
    
    def ShowEmojies(self):
        '''
        Function that show the GUI's Emojies used to represent the mood.
        '''
        self.Frame_Happy.show(); self.Lbl_Happy.show()
        self.Frame_Sad.show(); self.Lbl_Sad.show()
        self.Frame_Angry.show(); self.Lbl_Angry.show()
        self.Frame_Surprised.show(); self.Lbl_Surprised.show()
        self.Frame_Reflexive.show(); self.Lbl_Reflexive.show()
        self.Frame_Normal.show(); self.Lbl_Normal.show()
    
    def HideEmojies(self):
        '''
        Function that hide the GUI's Emojies used to represent the mood.
        '''
        self.Frame_Happy.hide(); self.Lbl_Happy.hide()
        self.Frame_Sad.hide(); self.Lbl_Sad.hide()
        self.Frame_Angry.hide(); self.Lbl_Angry.hide()
        self.Frame_Surprised.hide(); self.Lbl_Surprised.hide()
        self.Frame_Reflexive.hide(); self.Lbl_Reflexive.hide()
        self.Frame_Normal.hide(); self.Lbl_Normal.hide()
        
if __name__ == '__main__':
        
    App = QtGui.QApplication(sys.argv)
    GUI = EmotionsRecognition()
    sys.exit(App.exec_())    

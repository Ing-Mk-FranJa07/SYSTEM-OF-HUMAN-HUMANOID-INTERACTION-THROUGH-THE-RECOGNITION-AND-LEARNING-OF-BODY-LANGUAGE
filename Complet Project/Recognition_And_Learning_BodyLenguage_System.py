'''
Created on 14/09/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This is the main script of the system created to recognize and "teach" body 
language to Robot Pepper. 

The system use the Kinect camera to get the skeleton tracking and compute the 
orientation of the body, with this information is possible determined the mood 
of the person and also control the Robot Pepper.

The system allow classify the mood of a person in six categories: 
Happy, Sad, Angry, Surprised, Reflexive and Normal, are use emojies to
represent the category of the mood. Also this system controls the Robot Pepper 
playing motion sequences (Animations) coherent with the mood determined. 

The system has been developed to interact with the Robot Pepper through conversations
in real time, and playing motion sequences coherent with the conversation.

With this script is possible to make that the Robot Pepper imitate the body posture 
of one human in real time.

The system has a mode which try that Pepper "create" its owns motions sequences
using examples provides to the person in front the kinect camera.

Also, is possible "teach" to the Robot Pepper, new motion sequences using the 
imitation mode with a time limit.
'''
#coding: iso-8859-1

# Are imported the wrappers.
import time
import cv2
import numpy as np
from PyQt4 import uic, QtCore, QtGui                                            # Wrapper that allow use the GUI.
import Wrappers.Emojies_rc                                                      # Wrapper created to control the GUI's image.
import Wrappers.Kinect_Tracking as Sensor                                       # Wrapper created to use the Kinect V2 Camera.
import Wrappers.Pepper_Control as Robot                                         # Wrapper created to control basic functions of Pepper robot.
import Wrappers.Neural_Networks_Models as Network                               # Wrapper created to implemented Neural Networks models created previously.
import Wrappers.Save_Animation as Save                                          # Wrapper created to save new motion animation sequence.
import speech_recognition as sr                                                 # Wrapper that allow the Speech Recognition using PC's microphone.
import pygame
from pygame import surfarray
from pygame.time import Clock
import sys

try:                                                                            # Is created a function to show correctly the text in the GUI.
    Encoding = QtGui.QApplication.UnicodeUTF8
    def Translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, Encoding)
except AttributeError:
    def Translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

class Recognition_And_Learning_BodyLenguage_System(QtGui.QWidget):
    '''
    Main class of the system that allow recognize and control Pepper's body to
    express "emotions" and complement the speech in a coherent way with the 
    mood of the person.
    '''
    def __init__(self):
        super(Recognition_And_Learning_BodyLenguage_System, self).__init__()
        
        # Is "imported" the file that get the GUI.                
        self.MyGUI = uic.loadUi('...\Recognition_And_Learning_BodyLenguageGUI.ui', self)
        
        # Are created timers to have interruptions that allow execute functions.
        self.TimerKinect = QtCore.QTimer(self.MyGUI)                            # Timer that control the image tracking.
        self.TimerKinect.timeout.connect(self.Tracking)                         # The timer active the Tracking function.
        
        self.TimerSpeech = QtCore.QTimer(self.MyGUI)                            # Timer that control the speech recognition.
        self.TimerSpeech.timeout.connect(self.GetAudio)                         # The timer active GetAudio function.
        
        self.TimerAlternative = QtCore.QTimer(self.MyGUI)                       # Timer that control the "Alternative World" mode.        
        self.TimerAlternative.timeout.connect(self.Tracking)                    # The timer active Tracking function, in the option "Alternative World"
        
        self.TimerNewAnimation = QtCore.QTimer(self.MyGUI)                      # Timer that control the creation of new motion animation.        
        self.TimerNewAnimation.timeout.connect(self.Tracking)                   # The timer active Tracking function, in the option "New Animation".
        
        # Are associated the different options that have the GUI with its corresponding functions.
        self.connect(self.Btn_Close, QtCore.SIGNAL("clicked()"), self.EndProgram)
        self.connect(self.Btn_CreateNewAnimation, QtCore.SIGNAL("clicked()"), self.CreateAnimation)
        self.connect(self.Btn_ConectKinect, QtCore.SIGNAL("clicked()"), self.InitKinect)
        self.connect(self.Btn_ConectPepper, QtCore.SIGNAL("clicked()"), self.ConnectionPepper)
        self.connect(self.Btn_SaveNewAnimation, QtCore.SIGNAL("clicked()"), self.SaveNewAnimation)
        self.connect(self.CheckBox_Alternative, QtCore.SIGNAL("clicked()"), self.AlternativeWorld)
        self.connect(self.CheckBox_Imitation, QtCore.SIGNAL("clicked()"), self.ImitationMode)
        self.connect(self.CheckBox_Speech, QtCore.SIGNAL("clicked()"), self.SpeechRecognition)
        
        # Is created a pixel map to put in the GUI the image get with the Kinect V2 Camera.
        self.pixmap = QtGui.QPixmap()
        
        self.StartEndFlag = 1                                                   # Flag to start or end the Kinect V2 Camera.
        self.ImitationFlag = 0                                                  # Flag to start or end the game "Imitation mode".
        self.AlternativeFlag = 0                                                # Flag to start or end the game "Alternative World"
        self.CreteOrDelateFlag = 1                                              # Flag to create or delete a new motion animation sequence.
        self.SaveAnimationFlag = 0                                              # Flag to start or end the option "Create a new motion animation".    
                                                      
        self.MotionSequence = np.empty([39, 15])                                # List to save motion sequence use in game "Alternative World" and also in option "Save New Animation".
        self.NewAnimation = np.empty([39, 16])                                  # List to save the new motion animation created by the user.
        
        self.RecordingSec = 0                                                   # Counter that end the recording.

        self.DetectSpeech = sr.Recognizer()                                     # Allow use the pc's microphone.
        
        self.Kinect = Sensor.Kinect_Tracking()                                  # Instantiate the sensor wrapper's main class.
        self.NeuralNet = Network.Neural_Networks_Models()                       # Instantiate the network wrapper's main class.    
        
        self.MyGUI.show()

    def CreateAnimation(self):
        '''
        Function that active the option to create, play and save a new motion 
        animation sequence to complement the Pepper's animation database.
        '''
        if self.CreteOrDelateFlag == 1:
            self.Frame_Games.setEnabled(False)                                  # Disable games options.
            self.CheckBox_Speech.setEnabled(False)                              # Disable conversation option.
            
            if self.CheckBox_Alternative.isChecked():                           # Deactivate the game "Alternative World"
                self.CheckBox_Alternative.setCheckState(0)
                self.AlternativeWorld()
            
            elif self.CheckBox_Imitation.isChecked():                           # Deactivate the game "Imitation mode"
                self.CheckBox_Imitation.setCheckState(0)
                self.ImitationMode()
            
            self.ShowEmojies()                                                  # Show the emojies.    
            self.TimerKinect.stop()                                             # Deactivate the interruption to end the main mode.
            
            self.SaveAnimationFlag = 1                                          # Change flag's value to start the recording.
            
            # Is changed the text in the GUI button, and disable it.
            self.Btn_CreateNewAnimation.setText("Recording new motion animation sequence") 
            self.Btn_CreateNewAnimation.setEnabled(False)
            
            time.sleep(4)
            
            self.RecordingSec = 0
            self.SkeletonTracked = 0
            self.TimerNewAnimation.start(100)                                   # Activate the interruption to start the option "Create Animation".
        else:
            self.NewAnimation = np.empty([39, 16])                              # Clear the new motion animation sequence list.
            
            self.SaveAnimationFlag = 0                                          # Change flag's value to end the recording.
            self.CreteOrDelateFlag = 1                                          # Change the flag's value to create a new motion animation sequence.
            
            self.Btn_SaveNewAnimation.setEnabled(False)                         # Disable save new motion animation, option.    
            self.CheckBox_Imitation.setEnabled(True)                            # Enable the game "Imitation mode" option.
            self.CheckBox_Alternative.setEnabled(True)                          # Enable the game "Alternative world" option.
            self.CheckBox_Speech.setEnabled(True)                               # Enable conversation option.    
            
            # Is changed the text in the GUI button.
            self.Btn_CreateNewAnimation.setText("Create new motion animation sequence")
            
            self.HideEmojies()                                                  # Hide the emojies.
            self.TimerKinect.start(100)                                         # Activate the interruption to start the main option.

    def InitKinect(self):
        '''
        Function that start or End the Kinect V2 Camera.
        '''
        # Connect Kinect.
        if self.StartEndFlag == 1:
            # Are enable the options that allow the current games: Imitations mode, Alternative world.
            self.CheckBox_Imitation.setEnabled(True)
            self.CheckBox_Alternative.setEnabled(True)
            self.Btn_CreateNewAnimation.setEnabled(True)
            
            pygame.init()
            self.Clock = Clock()                                                # Frames/sec.
            
            self.Kinect.Open_Kinect()                                           # Call the function that start the kinect.
            
            self.Btn_ConectKinect.setText("Disconnect Kinect")                  # Change the text in the GUI button.
            
            self.HideEmojies()                                                  # Hide the representations of the emotions in the GUI.
            
            self.StartEndFlag = 0                                               # Change flag's value to end the Kinect V2 Camera.
            
            self.TimerKinect.start(100)                                         # Active the interruption that show the image get with the Kinect V2 Camera.
        
        # Disconnect Kinect.
        else:
            # Are disabled the options that allow the current games: Imitations mode, Alternative world.
            self.CheckBox_Imitation.setEnabled(False)
            self.CheckBox_Alternative.setEnabled(False)
            self.Btn_CreateNewAnimation.setEnabled(False)

            if self.CheckBox_Alternative.isChecked():                           # Deactivate the game "Alternative World"
                self.CheckBox_Alternative.setCheckState(0)
                self.AlternativeWorld()
            
            elif self.CheckBox_Imitation.isChecked():                           # Deactivate the game "Imitation mode"
                self.CheckBox_Imitation.setCheckState(0)
                self.ImitationMode()
                
            self.TimerKinect.stop()                                             # Deactivate the interruption that show the image get with the Kinect V2 Camera.
            
            self.Kinect.Stop_Kinect()
            
            pygame.quit()
            
            self.Btn_ConectKinect.setText("Connect Kinect")                     # Change the text in the GUI button.
            
            self.ShowEmojies()                                                  # Show the representations of the emotions in the GUI.
            
            self.StartEndFlag = 1                                               # Change the value flag's value to start the Kinect V2 Camera.
 
    def ConnectionPepper(self):
        '''
        Function that get the Ip and Port number to start the connection with Pepper.
        '''
        try:
            # Are get the Ip and Port numbers using the GUI.
            Ip = str(self.Text_IP.text())
            Port = int(self.Text_Port.text())
            
            self.Pepper = Robot.Pepper_Control(Ip, Port)                        # Instantiate the robot wrapper's main class.
            
            self.Frame_Conection.setEnabled(False)                              # Disable the text_box to Connect with Pepper. 
    
            # Are enabled the system options.
            self.Btn_ConectKinect.setEnabled(True)
            self.CheckBox_Speech.setEnabled(True)
                      
        except (RuntimeError, ValueError):                                      # Error alerts.
            if str(self.Text_Port.text()) == '':
                self.Text_Port.setText("Empty!")
            else:
                self.Text_Port.setText("Wrong!")
            if str(self.Text_IP.text()) == '':
                self.Text_IP.setText("Empty!")
            
            pass
    
    def SaveNewAnimation(self):
        '''
        Function that create a .csv file with the information of the joint angles
        that conform a motion animation sequence.
        ''' 
        # Is changed the text in the GUI button.
        self.Btn_SaveNewAnimation.setText("Saving new motion animation sequence")
        
        #time.sleep(2)
        
        Save.CreateFile(self.NewAnimation)                                      # Function that create a .csv file.
        
        # Are changed the text in the GUI buttons.
        self.Btn_SaveNewAnimation.setText("New motion animation sequence saved")
        self.Btn_CreateNewAnimation.setText("Create new motion animation sequence")
        
        #time.sleep(2)
        
        # Are changed the text in the GUI button, and disable it.
        self.Btn_SaveNewAnimation.setText("Save new motion animation sequence")
        self.Btn_SaveNewAnimation.setEnabled(False)                                 
        
        self.CheckBox_Imitation.setEnabled(True)                                # Enable the game "Imitation mode" option.
        self.CheckBox_Alternative.setEnabled(True)                              # Enable the game "Alternative world" option.
        self.CheckBox_Speech.setEnabled(True)                                   # Enable conversation option.    
            

        self.SaveAnimationFlag = 0                                              # Change flag's value to end the recording.
        self.CreteOrDelateFlag = 1                                              # Change the flag's value to create a new motion animation sequence.
        
        self.TimerKinect.start(100)
        
    def ImitationMode(self):
        '''
        Function that activate or deactivate the game "Imitation mode".
        '''
        if self.CheckBox_Imitation.isChecked():                                 # Verify is the option is select
            self.CheckBox_Imitation.setText("Imitation mode Activated")         # Change the text in the GUI check box.
            
            self.Pepper.InitPosition()                                          # Send Pepper to init position.
            
            self.ImitationFlag = 1                                              # Change the flag's value to start the game.
            
            if self.CheckBox_Alternative.isChecked():                           # Deactivate the game "Alternative World"
                self.CheckBox_Alternative.setCheckState(0)
                self.AlternativeWorld()
            
            self.ShowEmojies()                                                  # Show the emojies                                   
        
        else:
            self.CheckBox_Imitation.setText("Imitation mode Deactivated")       # Change the text in the GUI check box.
            
            self.Pepper.InitPosition()                                          # Send Pepper to init position. 
            
            self.ImitationFlag = 0                                              # Change the flag's value to end the game.
            
            self.HideEmojies()                                                  # Hide the emojies.
    
    def AlternativeWorld(self):
        '''
        Function that activate or deactivate the game "Alternative World".
        '''
        if self.CheckBox_Alternative.isChecked():                               # Verify is the option is select
            self.CheckBox_Alternative.setText('"Alternative World" Activated')  # Change the text in the GUI check box.
            
            self.Pepper.InitPosition()                                          # Send Pepper to initial position.
            
            self.AlternativeFlag = 1                                            # Change the flag's value to start the game.
            
            if self.CheckBox_Imitation.isChecked():                             # Deactivate the game "Imitation mode"
                self.CheckBox_Imitation.setCheckState(0)
                self.ImitationMode()
            
            self.TimerKinect.stop()                                             # Deactivate the interruption to end the main option.
            self.ShowEmojies()                                                  # Show the emojies 
            
            self.RecordingSec = 0
            self.SkeletonTracked = 0
            self.TimerAlternative.start(100)                                    # Active the interruption to start the option "recording"
        
        else:
            self.CheckBox_Alternative.setText('"Alternative World" Deactivated')# Change the text in the GUI check box.
                
            self.Pepper.InitPosition()                                          # Send Pepper to init position. 
            
            self.AlternativeFlag = 0                                            # Change the flag's value to end the game.
            
            self.TimerAlternative.stop()                                        # Deactivate the interruption to end the option "recording"
            
            self.HideEmojies()                                                  # Hide the emojies.
            self.TimerKinect.start()                                            # Activate the interruption to start the main option.

    def SpeechRecognition(self):
        '''
        Function that allow have a conversation with Pepper.
        This option deactivate the others functions of the system.
        '''
        if self.CheckBox_Speech.isChecked():                                    # Verify is the option is select
            self.CheckBox_Speech.setText("Talk with Pepper Activated")          # Change the text in the GUI check box.
                        if self.CheckBox_Alternative.isChecked():                           # Deactivate the game "Alternative World"
                self.CheckBox_Alternative.setCheckState(0)
                self.AlternativeWorld()
            
            elif self.CheckBox_Imitation.isChecked():                           # Deactivate the game "Imitation mode"
                self.CheckBox_Imitation.setCheckState(0)
                self.ImitationMode()
            
            else:
                self.Pepper.InitPosition()                                      # Send Pepper to initial position.
                
            self.StartEndFlag = 0                                               # Deactivate the tracking and Kinect V2 Camera.
            try:
                self.InitKinect()
            except:
                self.StartEndFlag = 1
                pass
            
            self.Btn_ConectKinect.setEnabled(False)                             # Disable the option to use the Kinect V2 Camera.
            
            self.CheckBox_Imitation.setEnabled(False)                           # Disable the game "Imitation mode" option.
            self.CheckBox_Alternative.setEnabled(False)                         # Disable the game "Alternative world" option.
                                          
            self.TimerSpeech.start(5)                                           # Activate the interruption to start the option "conversation".    
        
        else:
            self.CheckBox_Speech.setText("Talk with Pepper deactivated")        # Change the text in the GUI check box.

            self.TimerSpeech.stop()                                             # Deactivate the interruption to end the option "conversation".

            self.Btn_ConectKinect.setEnabled(True)                              # Enable the option to use the Kinect V2 Camera.
            
    def Tracking(self):
        '''
        Function that allow the tracking and get the body's joint angles to determine
        mood and control directly the Pepper's joints.
        '''   
        
        # ---------------------------------------------------------------------
            
        if self.AlternativeFlag == 1:                                           # Game: "Alternative World"
            self.RecordingSec += 1                                                                                    
            
            Surface, SequenceRec = self.Kinect.RunKinect(str('Pepper'))         # Get Kinect's image and an array with 15 joint angles.
            
            if SequenceRec is not None:
                self.MotionSequence[self.RecordingSec - 1] = SequenceRec        # Form a 39x16 joint angles matrix.
                self.SkeletonTracked += 1
                
            if self.RecordingSec == 39:                                         # Stop the recording.
                self.TimerAlternative.stop()                                    # Deactivate the interruption to end the option "recording".
                
                if self.SkeletonTracked > 25:
                    Animation = self.NeuralNet.GenerateAnimation(self.MotionSequence) # "Create" an original motion animation sequence.
                                        
                    self.Pepper.SendAnimation(Animation)                        # Play the motion animation sequence.
        
                self.MotionSequence = np.empty([39, 15])                        # Clear the matrix.
                self.RecordingSec = 0
                self.SkeletonTracked = 0
                                                         
                self.TimerAlternative.start(100)                                # Active the interruption to start the option "recording".
        
        # ---------------------------------------------------------------------
        
        elif self.ImitationFlag == 1:                                           # Game: "Imitation Mode"
            Surface, Sequence = self.Kinect.RunKinect(str('Pepper'))            # Get Kinect's image and an array with 15 joint angles.
            
            if Sequence is not None:                                            
                self.Pepper.SendSequence(Sequence)                              # Play motion sequence.
        
        # ---------------------------------------------------------------------

        elif self.SaveAnimationFlag == 1:                                       # Create new motion sequence.
            self.RecordingSec += 1
            
            Surface, CreateSequence = self.Kinect.RunKinect(str('Pepper'))      # Get Kinect's image and an array with 15 joint angles.
            
            if CreateSequence is not None:
                self.MotionSequence[self.RecordingSec - 1] = CreateSequence     # Form a 39x16 joint angles matrix.
                self.SkeletonTracked += 1
                
            if self.RecordingSec == 39:                                         # Stop the recording.
                self.TimerNewAnimation.stop()                                   # Deactivate the interruption to end the option "Create Animation".
                self.RecordingSec = 0
                
                if self.SkeletonTracked == 39:
                    self.CreteOrDelateFlag = 0                                  # Change the flag's value to delete the new motion animation created.
                    
                    # Is verified the new animation created, sending it to Pepper.                          
                    self.Btn_CreateNewAnimation.setText("Playing new motion animation sequence")
                    self.NewAnimation = Save.VerifyAndSenSecuence(self.MotionSequence, self.Pepper)
                                   
                    # Are enabled the options to the user: Delete new animation or Save new animation.
                    self.Btn_CreateNewAnimation.setText("Delete new motion animation sequence")
                    self.Btn_CreateNewAnimation.setEnabled(True)
                    self.Btn_SaveNewAnimation.setEnabled(True)

                    self.SkeletonTracked = 0
                    self.MotionSequence = np.empty([39, 15])                    # Clear the matrix.
                
                else:
                    self.Btn_CreateNewAnimation.setText("Recording data again") # Inform to the user that the data was not enough. 
                    self.CreteOrDelateFlag = 1                                  # Change the flag's value to delete the new motion animation created.
                    
                    time.sleep(1.5)
                    
                    self.SkeletonTracked = 0
                    self.MotionSequence = np.empty([39, 15])                    # Clear the matrix.
                    
                    self.CreateAnimation()
        
        # ---------------------------------------------------------------------
                
        else:                                                                   # Main mode: Recognition of Emotions.
            Surface, Sequence = self.Kinect.RunKinect(str('Emotion'))           # Get Kinect's image and an array with 25 joint angles.
            
            if Sequence is not None:
                self.TimerKinect.stop()                                         # Deactivate the interruption that show the image get with the Kinect V2 Camera.
                                            
                self.ShowEmotion(Sequence)                                      # Determine mood.
                
                self.TimerKinect.start(100)                                     # Active the interruption that show the image get with the Kinect V2 Camera.
        
        # ---------------------------------------------------------------------        
        
        # Is reshape the Kinect's image in the format that allow cv2.
        Image = surfarray.array3d(Surface)                                      # Pygame's surface is converted in an array. 
        Image = np.rollaxis(Image, 0, 2)                                        # Reform the structure of the array.
         
        # Is reformat and reshape the Image to be put in the GUI.
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        Image = cv2.resize(Image,(1080,630), interpolation = cv2.INTER_CUBIC)
        
        Image = QtGui.QImage(Image, 1080, 630, 3240, QtGui.QImage.Format_RGB888)# Create a pixel map from the Kinect's image.
        self.pixmap.convertFromImage(Image.rgbSwapped())                        # Specify a RGB format.
        self.KinectFrame.setPixmap(self.pixmap)                                 # Show the Kinect's image in the GUI.
        
        self.Clock.tick(60)

    def ShowEmotion(self, Sequence):
        '''
        Function that show the mood of the person and do that Pepper acts consistently.
        '''
        EmotionCode = self.NeuralNet.RecognizeEmotions(Sequence)                # Function that determinate the mood, through a RNA.
        AnimationCode = np.random.randint(2)                                    # Random selector to determinate which animation Pepper will play.
        
        # ---------------------------------------------------------------------
        # The current lines show in the GUI the mood using emojies and send to 
        # Pepper, motion animations sequence which is determined by an index
        # that change in function of the "AnimationCode" random number,
        # and text to transform in speech consistently with the mood showed.
        # ---------------------------------------------------------------------   
        
        # Show Happy:
        if np.all(EmotionCode == np.array([0,0,0,0,0,1])): 
            self.HideEmojies()
            self.Frame_Happy.show(); self.Lbl_Happy.show()
            
            if AnimationCode == 0:
                Animation = self.NeuralNet.SelectAnimation(str("Victory"))
                self.Pepper.SendToPepperSpeech(str("Yeah! you did it!"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            else:
                Animation = self.NeuralNet.SelectAnimation(str("Yeah"))
                self.Pepper.SendToPepperSpeech(str("Oh Yeah, oh Yeah!"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)

        # Show Sad:
        elif np.all(EmotionCode == np.array([0,0,0,0,1,0])):
            self.HideEmojies()
            self.Frame_Sad.show(); self.Lbl_Sad.show()
            
            if AnimationCode == 0:
                Animation = self.NeuralNet.SelectAnimation(str("Why_sad"))
                self.Pepper.SendToPepperSpeech(str("Why are you sad?"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            else:
                Animation = self.NeuralNet.SelectAnimation(str("Don't_be_sad"))
                self.Pepper.SendToPepperSpeech(str("Come on don't be sad"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)

        # Show Angry
        elif np.all(EmotionCode == np.array([0,0,0,1,0,0])):
            self.HideEmojies()
            self.Frame_Angry.show(); self.Lbl_Angry.show()
            
            if AnimationCode == 0:
                Animation = self.NeuralNet.SelectAnimation(str("Scary"))
                self.Pepper.SendToPepperSpeech(str("You are so scary"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            else:
                Animation = self.NeuralNet.SelectAnimation(str("Calm_down"))
                self.Pepper.SendToPepperSpeech(str("Please! calm down"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            
        # Show Surprised
        elif np.all(EmotionCode == np.array([0,0,1,0,0,0])):
            self.HideEmojies()
            self.Frame_Surprised.show(); self.Lbl_Surprised.show()

            if AnimationCode == 0:
                Animation = self.NeuralNet.SelectAnimation(str("What_happened"))
                self.Pepper.SendToPepperSpeech(str("Oh, what happened!"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            else:
                Animation = self.NeuralNet.SelectAnimation(str("Whats_up"))
                self.Pepper.SendToPepperSpeech(str("What was that?"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            
        # Show Reflexive
        elif np.all(EmotionCode == np.array([0,1,0,0,0,0])):
            self.HideEmojies()
            self.Frame_Reflexive.show(); self.Lbl_Reflexive.show()
            
            if AnimationCode == 0:
                Animation = self.NeuralNet.SelectAnimation(str("What_are_thinking"))
                self.Pepper.SendToPepperSpeech(str("What are you thinking?"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
            else:
                Animation = self.NeuralNet.SelectAnimation(str("Are_somebody_there"))
                self.Pepper.SendToPepperSpeech(str("Hello! are somebody inside this head?"))
                time.sleep(1)
                self.Pepper.SendAnimation(Animation)
        
        # ----------------------------------------------------------------------
        # When the mood is classified like "Normal" is showed the emojie and 
        # is activated the speech recognition to talk with Pepper.
        # When the mood is not determined correctly, all the emojies are hidden.
        # ----------------------------------------------------------------------
            
        # Show Normal:
        elif np.all(EmotionCode == np.array([1,0,0,0,0,0])):
            self.HideEmojies()
            self.Frame_Normal.show(); self.Lbl_Normal.show()
            
            # self.GetAudio()                                                     # Function that recognize the speech using pc's microphone.

        # Mood no determined:
        else:
            self.HideEmojies()        

    def GetAudio(self):
        '''
        Function that allow do the speech recognition getting the audio through 
        the pc's microphone and transforming it into a text, to send it at the LSTM 
        model use to predict the "answer" that Pepper goes to say. 
        '''        
        self.Lbl_Listening.setText(" ")
        self.Lbl_PrintAnswer.setText(" ")
          
        with sr.Microphone() as source:                                         # Activate the microphone.
            self.DetectSpeech.adjust_for_ambient_noise(source, duration = 3)    # Calibrate the microphone to filter environmental noise.
            Audio = self.DetectSpeech.listen(source)                            # Capture the audio data.
        
        # If the audio data was catch, is generate an "answer".
        try:
            Question = self.DetectSpeech.recognize_google(Audio)                # Transform the audio data to text.
            
            Answer, Animation = self.NeuralNet.GetAnswer(Question)              # Get the "answer", and the motion animation consistently with the Pepper's answer.
            
            self.Lbl_Listening.setText(Question)                                # Show in the GUI the speech recognized.
            self.Lbl_PrintAnswer.setText(str(Answer))                           # Show in the GUI the "answer".
            
            self.Pepper.SendToPepperSpeech(str(Answer))                         # Send to Pepper the "answer".
            time.sleep(1)
            self.Pepper.SendAnimation(Animation)                                # Send to Pepper the motion animation.
            
        # If the audio data was not catch correctly, the user is informed.
        except sr.UnknownValueError:
            self.Lbl_Listening.setText("No Listened !")
        except sr.RequestError as e:
            self.Lbl_Listening.setText('Audio error; {0}'.format(e))
            pass
    
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

    def EndProgram(self):
        '''
        Function that end the tracking and close the program, before send Pepper
        to initial position.
        '''
        try:
            self.Pepper.InitPosition()                                          # Send Pepper to initial Position.
    
            self.Kinect.Stop_Kinect()                                           # Stop Kinect V2 Camera.
            pygame.quit()                                                       # End visualization of kinect's image.

            sys.exit()                                                          # Close the GUI.    
        except:
            sys.exit()                                                          # Close the GUI.    
            
if __name__ == '__main__':
        
    App = QtGui.QApplication(sys.argv)
    GUI = Recognition_And_Learning_BodyLenguage_System()
    try:
        sys.exit(Robot.Pepper_Control.InitPosition())
    except:
        sys.exit(App.exec_())
    
    
    

    
    
        

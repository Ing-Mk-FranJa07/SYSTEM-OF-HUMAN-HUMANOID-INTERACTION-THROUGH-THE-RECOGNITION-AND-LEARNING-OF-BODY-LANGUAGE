'''
Created on 14/09/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

Wrapper that allow control the Robot Pepper. 

To go further review the API in the Softbank Robotics Documentation. NAOqi.
'''
# Is imported the library to control Pepper.
from naoqi import ALProxy
import almath
import time

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

class Pepper_Control(object):
    '''
    Wrapper class to control basic functions of robot Pepper like text to speech
    and also control of moving of Pepper's joints.
    '''
    def __init__(self, Ip, Port):
        self.Ip = Ip
        self.Port = Port

        self.Motion = ALProxy('ALMotion', self.Ip, self.Port)
        self.Posture = ALProxy('ALRobotPosture', self.Ip, self.Port)
        self.Speak = ALProxy('ALTextToSpeech', self.Ip, self.Port)
                
        self.Motion.wakeUp()
        self.Posture.goToPosture('Stand', 0.75)
    
    def SendToPepper(self, JointName, Angle):
        '''
        Function that do that Pepper put its joints in the specific orientation.
        '''
        self.Motion.setAngles(JointName, Angle, JointsParams[JointName]['fractionMaxSpeed'])

    def SendToPepperHands(self, Hand, State):
        '''
        Function that do that Pepper close and open its hands.
        '''
        if State == 1:
            self.Motion.setAngles(Hand, 1, JointsParams[Hand]['fractionMaxSpeed'])
        elif State == 2:
            self.Motion.setAngles(Hand, 0, JointsParams[Hand]['fractionMaxSpeed'])
        else:
            self.Motion.setAngles(Hand, 0.5, JointsParams[Hand]['fractionMaxSpeed'])

    def GetToPepper(self,JointName):
        '''
        Function that read the Pepper's joints sensors.
        '''
        AngleRad = self.Motion.getAngles(JointName, True)
        AngleRad = float(AngleRad[0])
        return AngleRad
    
    def SendToPepperSpeech(self, Speech):
        '''
        Function that send to Pepper the speech.
        '''
        self.Speak.say(Speech)
    
    def InitPosition(self):
        '''
        Function that send Pepper's joints to "init" position.
        '''
        self.Posture.goToPosture('StandInit', 1.0)
        
    def SendAnimation(self, Animation):
        '''
        Function that send to Pepper a Motion animation.
        '''
        self.InitPosition()                                                     # Send Pepper to initial position.

        for I in range(len(Animation)):                                         # Play Motion animation.
            # Send Hip angles.
            if Animation[I, 0] is not 0:
                self.VerifyHipAngles('HipPitch', Animation[I, 0])
            self.VerifyHipAngles('HipRoll', Animation[I, 1])
            
            # Send Head angle.
            if Animation[I, 2] is not 0:
                self.VerifyHeadYawAngles(Animation[I, 2])
            self.VerifyHeadPitchAngles(Animation[I, 3], Animation[I, 2])
    
            # Send Right Shoulder angles.
            self.VerifyShouldersAndWristsAngles('RShoulderPitch', Animation[I, 4])
            self.VerifyShouldersAndWristsAngles('RShoulderRoll', Animation[I, 5])
    
            # Send Right Elbow angles.
            self.VerifyRElbowYawAngles(Animation[I, 6])
            self.VerifyRElbowRollAngles(Animation[I, 7], Animation[I, 6])
    
            # Send Right Wrist angle.
            self.VerifyShouldersAndWristsAngles('RWristYaw', Animation[I, 8])
    
            # Send Right Hand state.
            self.Motion.setAngles('RHand', Animation[I, 9], JointsParams['RHand']['fractionMaxSpeed'])
    
            # Send Left Shoulder angles.
            self.VerifyShouldersAndWristsAngles('LShoulderPitch', Animation[I, 10])
            self.VerifyShouldersAndWristsAngles('LShoulderRoll', Animation[I, 11])
    
            # Send Left Elbow angles.
            self.VerifyLElbowYawAngles(Animation[I, 12])
            self.VerifyLElbowRollAngles(Animation[I, 13], Animation[I, 12])
    
            # Send Left Wrist angle.
            self.VerifyShouldersAndWristsAngles('LWristYaw', Animation[I, 14])
        
            # Send Left Hand state.
            self.Motion.setAngles('LHand', Animation[I, 15], JointsParams['LHand']['fractionMaxSpeed'])
            
            time.sleep(0.1)
    
    def SendSequence(self, Sequence):
        '''
        Function that send Pepper's each joint each.
        '''        
        if not None in Sequence:                                                    # Verify allow data.
             
          # Send Hip angles.
          self.VerifyHipAngles('HipPitch', Sequence[0][14])
          self.VerifyHipAngles('HipRoll', Sequence[0][13])

          # Send Head angle.
          self.VerifyHeadPitchAngles(Sequence[0][0], 0)

          # Send Right Shoulder angles.
          self.VerifyShouldersAndWristsAngles('RShoulderPitch', Sequence[0][2])
          self.VerifyShouldersAndWristsAngles('RShoulderRoll', Sequence[0][1])

          # Send Right Elbow angles.
          self.VerifyRElbowYawAngles(Sequence[0][3])
          self.VerifyRElbowRollAngles(Sequence[0][4], Sequence[0][3])

          # Send Right Wrist angle.
          self.VerifyShouldersAndWristsAngles('RWristYaw', Sequence[0][5])

          # Send Right Hand state.
          self.SendToPepperHands('RHand', Sequence[0][6])

          # Send Left Shoulder angles.
          self.VerifyShouldersAndWristsAngles('LShoulderPitch', Sequence[0][8])
          self.VerifyShouldersAndWristsAngles('LShoulderRoll', Sequence[0][7])

          # Send Left Elbow angles.
          self.VerifyLElbowYawAngles(Sequence[0][9])
          self.VerifyLElbowRollAngles(Sequence[0][10], Sequence[0][9])

          # Send Left Wrist angle.
          self.VerifyShouldersAndWristsAngles('LWristYaw', Sequence[0][11])

          # Send Left Hand state.
          self.SendToPepperHands('RHand', Sequence[0][12])
        
        else:
          return
        
    '''
    The current functions verify and limit the range of motion of each joint 
    to the range allowed by Pepper.
    ------------------------------------------------------------------------
    The options with "LSup" and "LInf", have been calculated through linear 
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
    
    
        


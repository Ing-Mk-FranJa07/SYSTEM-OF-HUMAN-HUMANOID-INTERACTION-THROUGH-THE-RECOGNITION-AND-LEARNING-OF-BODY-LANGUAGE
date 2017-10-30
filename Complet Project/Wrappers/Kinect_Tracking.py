'''
-------------------------------------------------------------------------------------------
# Based on the example "PyKinectBodyGame.py" provided by PyKinect2
# This code was released under the MIT license:
#
#    The MIT License (MIT)
#
#    Copyright (c) Microsoft
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
--------------------------------------------------------------------------------------------
# Relevant modifications to the base script made at HEIG-VD are highlighted and/or commented
# Developpers:
#  Julien Rebetez
#  Eric Henchoz
#  Hector Satizabal
# Institute IICT, HEIG-VD 2016
#
# This script employs the quaternions provided by the Kinect2 SDK. Each
# quaternion represents the absolute orientation of its parent bone, e.g.,
# the quaternion at the wrist represents the absolute orientation of the
# lower arm bone. In order to obtain orientations relative to a certain base,
# we pre-multiplied the quaternion by the inverse of the base quaternion,
# e.g., Qwrist-elbow = inv(Qelbow) * Qwrist. Relative orientations were then
# transformed into rotation matrices in order to obtain the Euler angles
# describing the 3 rotations: pitch, roll and yaw. We used the order 'syzx'
# which means that the rotations were done with respect to the static
# reference and they are applied in the following order: (1)pitch around x,
# (2)roll around z and (3)yaw around y. This order matchs the construction
# constaints of the robot. The sequence 'syzx' was empirically found by first
# making the two coordinate systems to match (moving the limbs and making
# both bones colinear), and then finding the sequence of axes around wich
# rotations has to be made.
----------------------------------------------------------------------------------------------
Created on 14/09/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

This wrapper get an image with the Kinect camera, the image is transformed into 
a RGB format in the main script to show it in the GUI. With the kinect is possible 
get the skeleton tracking, and compute the joint's angles to recognize the mood 
of the person and "teach" to the Robot Pepper body language, also is possible control
directly the Robot Pepper.
'''
# Are imported the wrappers that have the methods to get the skeleton joints angles.
import sys 
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import ctypes
from pygame import color, draw, Surface, surfarray
import numpy as np
import cv2
from math import pi
import transformations as tf

# Is defined a vector that has the parameters to draw the skeleton.
SkeletonColors = [color.THECOLORS["green"], color.THECOLORS["red"]] 

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

class Kinect_Tracking(object):
    '''
    Wrapper class to use the Kinect V2 Camera to get the skeleton tracking
    The current implementation return the eulerian angles (Pitch, Yaw, Roll) 
    of each bodie's joint: Head, Shoulders, Elbows, Wrist, Hip, Knee; and the 
    state of the hands (Open-Close)
    '''
    def __init__(self):
        self.Bodies = None
    
    def Open_Kinect(self):                              	                    # Manual Start of the Kinect V2 Camera.  
        self.Kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
        self.FrameSurface = Surface((self.Kinect.color_frame_desc.Width, self.Kinect.color_frame_desc.Height), 0, 32)
    
    def Stop_Kinect(self):                                                      # Manual Stop of the Kinect V2 Camera.
        self.Kinect.close()
    
    def DrawColorFrame(self, Frame, TargetSurface):
        '''
        Function that decode the image get with the Kinect V2 Camera and 
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
        Kinect V2 Camera and return the information of that body.
        '''
        NearestBody = None
        NearestDistance = float('inf')
     
        for i in range(0, self.Kinect.max_body_count):
            Body = self.Bodies.bodies[i] 
            if not Body.is_tracked: 
                continue
            
            Spine = Body.joints[PyKinectV2.JointType_SpineBase].Position        # Spine coordinates.
            Distance = np.sqrt((Spine.x**2)+(Spine.y**2)+(Spine.z**2))          # Compute the eucledian distance of the body to the Kinect V2 Camera.

            if Distance < NearestDistance:
                NearestDistance = Distance
                NearestBody = Body
                
        return NearestBody  

    def DrawBody(self, Joints, JointsPoints, Color):
        '''
        Function that send the parameters get with the Kinect V2 Camera
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

    def Angles(self, Joints, Orientations, Body, Type):
        '''
        Function that calculated the joint's angles of the skeleton get with the
        Kinect V2 Camera, getting the quaternion matrix and then calculating the
        eulerian angles.

        If the skeleton tracking is not correct, complete or parcial, the 
        corresponding angles are define like "None"; after that the angle are 
        verified, they are saved in the corresponding lists.
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
                                HandL = 1                                       # Hand opened.    
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
                
            if Type == 'Emotion':
                return np.transpose([[HeadPitch], 
                                     [RShoulderRoll], [RShoulderPitch], [RElbowYaw], [RElbowRoll], [RWristYaw], [HandR], 
                                     [LShoulderRoll], [LShoulderPitch], [LElbowYaw], [LElbowRoll], [LWristYaw], [HandL],
                                     [WaistRoll], [WaistPitch],
                                     [RHipRoll], [RHipPitch], [RKneeYaw], [RKneeRoll],
                                     [LHipRoll], [LHipPitch], [LKneeYaw], [LKneeRoll]]) 
            
            if Type == 'Pepper':
                return np.transpose([[HeadPitch], 
                                     [RShoulderRoll], [RShoulderPitch], [RElbowYaw], [RElbowRoll], [RWristYaw], [HandR], 
                                     [LShoulderRoll], [LShoulderPitch], [LElbowYaw], [LElbowRoll], [LWristYaw], [HandL],
                                     [WaistRoll], [WaistPitch]]) 
        
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
                draw.line(self.FrameSurface, Color[1], Start, End, 8)
            
            else:
                draw.line(self.FrameSurface, Color[0], Start, End, 8)
                
        except: 
            pass

    def Quaternion(self, Orientations, Joint, ParentJoint = None):
        '''
        Function that computes the quaternion matrix using the coordinates of
        each joint get with the Kinect V2 Camera.
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
    
    def RunKinect(self,  Type):
        '''
        Function that do the tracking process and draw the skeleton in a frame
        compatible with the cv2 format.
        '''        
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
                        AnglesSecuence = self.Angles(Joints, Orientations, Body, Type)
                        
                        return self.FrameSurface, AnglesSecuence
                    
                    else:
                        return self.FrameSurface, None
                
                else:
                    return self.FrameSurface, None
            
            else:
                return self.FrameSurface, None
        
        else:
            return self.FrameSurface, None


    
    
    
        
    
    
    







# Body posture data base creator.

### Description:

The script developed [DataBaseCreatorHumanPosture.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Data%20Base%20Creator/DataBaseCreatorHumanPosture.py) works using the GUI **DataBaseCreatorHumanPostureGUI.ui** to allow to the user create files that contain the information of the each joint orientation of the body that describe an human emotion. 

The system use the **Microsoft Kinect V2 camera** to get the skeleton tracking and the body angles orientation in the format (Yaw, Roll, Pitch) of the posture that the user is performing. The user can select the type of posture that he wants represent between eight classes: Happy, Sad, Angry, Surprised, Reflexive, Insecure, Trusted and Normal. The system allows to the user create a .csv file with the information and also a .avi file with the video of his performing. 

The .csv files created can be used to train and test an [Artificial Neural Network (ANN)](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) used to recognize the mood of the humans, implemented in the [System to interact with an Humanoid Robot](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project) and also in a beta system used to detect and [recognize the mood and the speech recognition](
https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Recognition).

* The image shows a graphical representation of the system described.
![body posture data base creator graphical process](https://user-images.githubusercontent.com/31509775/33278447-905b35a4-d369-11e7-96a5-0b8f3098a2bc.PNG)

### Hardware and software requirements:

* Microsoft Kinect V2 sensor.

![kinect](https://user-images.githubusercontent.com/31509775/32930198-222ed504-cb2b-11e7-8455-ba7d30df2631.jpg)

* Microsoft Kinect V2 adapter. 

![kinect adapter](https://user-images.githubusercontent.com/31509775/32930206-2a22a600-cb2b-11e7-86f9-96ecb8669ddc.jpg)

This system was developed using **PYTHON 2.7 (32 bits) in WINDOWS 10** to run correctly this script is necessary first to have installed:

Is recommended install [Anaconda (Python 2.7 (32 bits) version](https://www.anaconda.com/download/#windows) to get easier the packages necessaries. 

* [Kinect for Windows SDK version 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561): Microsoft website link with the installation instructions.
* [pykinect2](https://github.com/Kinect/PyKinect2): GitHub link, this repository has all the instructions to use the Kinect V2 with Python.
* [PyQT4 GPL version 4.11.4 for Python 2.7 (32 bits)](https://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe/download): Direct downlad link.
* [pygame version 1.9.2](http://www.pygame.org/news): pygame website link, you can found the download option for the pygame version 1.9.2 there.
* [pandas version 0.19.2](https://pandas.pydata.org/): pandas website link, the download option to the version 0.19.2 is there.
* [cv2 version 3.0.0](https://docs.opencv.org/3.3.1/d5/de5/tutorial_py_setup_in_windows.html): OpenCV website link.
* numpy version 1.12.1.
* ctypes version 1.1.0

### WARNINGS:

* The line 120 allows load the GUI created to use the system **Please make sure that the path of the file is correct !**

```python
   119        # Is loaded the file that get the GUI.   
   120        self.MyGUI = uic.loadUi('...\Emotions Data Base Creator\DataBaseCreatorHumanPostureGUI.ui', self)
```

* The line 349 saves the name of the .csv file and the location used to save the file **Please make sure that the path of the file is correct !**

```python
   348        # Is saved the name of the .csv file.
   349        FileName = str("...\Emotions Data Base Creator\Motion sequences\ " + str(self.Text_Save.text()) + ".csv")
```

* The lines 453-466 allow load the .avi and .csv files **Please make sure that the path of the .csv file (line 455) and the .avi dile (line 459) are the same and correct !**

```python
   452        # Are loaded the .avi and .csv files
   453        try: 
   454            # .csv file.
   455            FileName = str("...\Emotions Data Base Creator\Motion sequences\ " + str(self.Text_Load.text()) + ".csv")
   456            File = pd.read_csv(FileName, header = 0)
   457
   458           # .avi file.
   459           VideoName = str("...\Emotions Data Base Creator\Motion sequences\ " + str(self.Text_Load.text()) + ".avi")
   500           self.Video = cv2.VideoCapture(VideoName)
   501
   502           FileLoad = 1                                                        # Check the correct load.
   503            
   504        except:
   505           FileLoad = 0                                                        # Check the incorrect load.
   506           pass
```

* The line 689 allows create the .avi file with the name digited by the user **Please make sure that the path is correct and the same of the line 349 !**

```python
   688        # Is saved the name of the .avi file
   689        VideoFilePath =  str("...\Emotions Data Base Creator\Motion sequences\ " + str(self.Text_Save.text()) + ".avi")
```

**The author used this system to create [the body posture used to train the ANN](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Emotions%20Data%20Base%20Creator/Emotions%20DataBase) if you want reproduce this posture, it wont work because there are not in the repository the .avi files** 

### Code explanation:

The system works using the GUI developed in whic the user can record a new body posture using the kinect camera and selecting the type of the emotion that he is going to represent. Digiting the name which he wants save the information the user can create a .avi and .csv files after that the system has recorded the posture. 

When the system started, the GUI is thrown and the user can connect the kinect to create a new posture or loaded a posture created digiting the name of the .avi and .csv files and loaded them. If the user wants create a new posture he must connect the kinect and select the type of the emotion that he goes to perform with his body posture, after that the user must to put the name of to save the files and then he can record the posture and finally, if he wants, he can save the informacion (video and angles data) or restart the process.

* The image shows the general flowchart of the system.
![flowchart body posture data base creator general process](https://user-images.githubusercontent.com/31509775/33280062-dd34a388-d36e-11e7-86f8-dbcce559e451.PNG)

***Record a new body posture***

The system get back the name of the .avi and .csv file digited by the user, the lists used to save the angles orientation of the body joints are cleared, and finally start the recordign, saving the image gotten by the kinect and computing the skeleton tracking to come back the eulerian angles orientation while three seconds.

* The image shoes the flowchart to the record a motion sequence process.
![flowchart body posture data base creator create new body posture](https://user-images.githubusercontent.com/31509775/33280180-3cf784ca-d36f-11e7-93e5-adb35a5ee0ac.PNG)

The ***skeleton tracking*** process start to save, and then show in the GUI, the new color frame catching by the Kinect and searching for a body in it. The kinect can recognize until six bodies at the same time, so just the information of the nearest body is used. The kinect can compute the spatial points and the orientation of each bodi joint, so the next step is come back and save this information from the kinect and draw the skeleton representation on the user body in the image.

* The image shows the flowchart of the skeleton tracking process.
![flowchart body posture data base creator skeleton tracking process](https://user-images.githubusercontent.com/31509775/33280233-64a25c16-d36f-11e7-9144-18885f7a71aa.PNG)

To show image gotten by the Kinect to the user using the GUI, is necessary create a surface to "paste" the image in it, and then create a RGB image from the surface. Is necessary modify the structure of the matrix resulting to generate a 3-D matrix with the RGB format and the size of the GUI frame. Finally must be created a pixel map from the RGB image to be "printed" in the GUI frame.

* The image shows the flowchart of the show image process.
![flowchart body posture data base creator show image](https://user-images.githubusercontent.com/31509775/33280261-79adb132-d36f-11e7-9ccf-0dc7a0745e8f.PNG)

To draw the skeleton representation on the user's body in the image, is necessary use the spatial coordinates of two adjacent joints, for example, to draw the head is necesary use the head and the neck spatial coordinates, to draw the right arm is necessary use the right shoulder and right elbow spatial coordinates. 

* The image shows the logical process to draw the skeleton tracking.
![flowchart body posture data base creator draw bones](https://user-images.githubusercontent.com/31509775/33280289-952d7334-d36f-11e7-8bf7-09aaeaac2706.PNG)

The process to draw the "bones"of the body is a simple process that verify if each joint have been tracked correctly or not, and use the spatial coordinates (x, y) from the first joint to start the bone and the coordiantes of the second joint to end the bone. Is the both joints were tracked correctly the bone will be drawn with a green color, if one of them was not tracked correctly the bone will be red.

* The image shows the flowchart to the process to draw the bones.
![flowchart body posture data base creator draw bones](https://user-images.githubusercontent.com/31509775/33280371-d369d67e-d36f-11e7-9a81-6d16fd02e2b5.PNG)

To get the eulerian angles (Yaw, Roll, Pitch) is necessary verify if the respective joint have been tracked correctly, if this the case, is get back the quaternion that contain the orientation and then is calculate the eulerian angles from it; is the joint was not tracked correctly its eulerian angles are saved with the "none" value. Are saved 23 angles, the process begin with the computing of the waist angles to guarantee tha have been done the skeleton tracking, and then are computed all the necessaries angles of each joint following a logic process.

* The image shows the flowchart to the computing eulerian angles process.
![flowchart body posture data base creator get eulerian angles process](https://user-images.githubusercontent.com/31509775/33280402-f15a9754-d36f-11e7-9871-1eb5bdc98da0.PNG)

The angles are verify to delete any "none" value, if is found a none value this is replaced by the previous correct value. The control of the body start sending the waist angles values, and then the head angles values, next is sent the right arm angles values and finally the left arms angles values. All angles values must to be verify before to be sent, because the robot has a limitations in its joints moves ranges.

* The image shows the flowchart to verify the "none" values.
![flowchart body posture data base creator verify none values](https://user-images.githubusercontent.com/31509775/33280942-a906a72a-d371-11e7-8852-9f49a872f34d.PNG)

Finally the angles are saved in the .csv file created.

***Load and reproduce a body posture***

To load an animation created and saved previously, the system come back the name of the .avi and .csv files digited by the user, and clear the lists used to have the angles information to each joint. The files are loaded and the angles information is saved in the corresponding lists, if the .avi file was loaded succesfull, the video and the angles information are playing at the same time, show them in the GUI. 

* The image shows the flowchart of the load animation process.
![flowchart body posture data base creator load a body posture](https://user-images.githubusercontent.com/31509775/33281012-d0351516-d371-11e7-8bc9-3489fb279a96.PNG)

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

### Code explanation:

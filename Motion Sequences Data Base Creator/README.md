# Animation (motion sequences) data base creator.

### Description:

The script developed [AnimationDataBaseCreator.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Motion%20Sequences%20Data%20Base%20Creator/AnimationDataBaseCreator.py) works using the GUI created **AnimationDataBaseCreatorGUI.ui** to allow to the user created motion sequences used to be reproduce by the [Robot Pepper](https://www.ald.softbankrobotics.com/en/robots/pepper). This sequences will be used to the robot interact with the persons performing a coherent behavior with the mood of the persons and the conversation that is having with them, using the system [Recogniton and learning body lenguage system](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project) Also the data base created with this tool, can be used to train the [Genarative Adversarial Network (GAN) model built](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Create%20Motion%20Sequences) to increase the data base of the animations and give to the robot original animations developed by the network.

The system works usign the **Microsoft Kinect V2 Camera** to get the skeleton tracking and compute the angles orientation in the format (Yaw, Roll, Pitch) of each joint; using this system the user can perfom a motion sequence with his body and then this sequence goes to be reproduce by the robot, "teaching" to the robot how to behave in different situations. The user can get the feedback of his moves in the image showed in the GUI. The animation can ve saved in a .csv file after have been reproduce by the robot. The system create a .avi file with the images of the user performing the motion sequence.

With the system the user can load and reproduce the video and the animations saved previously, also it can be reproduce the [motion sequences created by the GAN model](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Create%20Motion%20Sequences/DataBaseGeneratedByRNA).

* The images shows a graphical representation of the system described.
![animation data base creator system](https://user-images.githubusercontent.com/31509775/33232082-cd1405d6-d1cd-11e7-98a8-323f906b096e.PNG)

### Hardware and software requirements:

* Microsoft Kinect V2 sensor.

![kinect](https://user-images.githubusercontent.com/31509775/32930198-222ed504-cb2b-11e7-8455-ba7d30df2631.jpg)

* Microsoft Kinect V2 adapter. 

![kinect adapter](https://user-images.githubusercontent.com/31509775/32930206-2a22a600-cb2b-11e7-86f9-96ecb8669ddc.jpg)

* Robot Pepper developed by Aldebaran robotics from SoftBank group. 

![robot peppper](https://user-images.githubusercontent.com/31509775/33133904-3c61430e-cf6c-11e7-9e78-a23be64623ab.png)

This system was developed using **PYTHON 2.7 (32 bits) in WINDOWS 10** to run correctly this script is necessary first to have installed:

Is recommended install [Anaconda (Python 2.7 (32 bits) version](https://www.anaconda.com/download/#windows) to get easier the packages necessaries. 

* [Coregraphe suit version 2.5.5 from SoftBank Robotics Community](http://doc.aldebaran.com/2-5/software/choregraphe/installing.html#desktop-installation): Aldebaran documentation website. Choregraphe suit installation guide.
* [pynaoqi version 2.5.5.5](http://doc.aldebaran.com/2-5/dev/python/install_guide.html#python-install-guide): Aldebaran documentation website. Python SDK installation guide.
* [Kinect for Windows SDK version 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561): Microsoft website link with the installation instructions.
* [pykinect2](https://github.com/Kinect/PyKinect2): GitHub link, this repository has all the instructions to use the Kinect V2 with Python.
* [PyQT4 GPL version 4.11.4 for Python 2.7 (32 bits)](https://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe/download): Direct downlad link.
* [pygame version 1.9.2](http://www.pygame.org/news): pygame website link, you can found the download option for the pygame version 1.9.2 there.
* [pandas version 0.19.2](https://pandas.pydata.org/): pandas website link, the download option to the version 0.19.2 is there.
* [cv2 version 3.0.0](https://docs.opencv.org/3.3.1/d5/de5/tutorial_py_setup_in_windows.html): OpenCV website link.
* numpy version 1.12.1.
* ctypes version 1.1.0

### WARNINGS:

* The line 149 allows load the GUI developed to use the system. **Plase make sure that the path of the file is correct !**

```python
   148        # Is loaded the file that get the GUI.  
   149        self.MyGUI = uic.loadUi('...\DataBaseCreatorSecuenceOfMovements\AnimationDataBaseCreatorGUI.ui', self)
```

* The line 263 allows get back the name digited by the user to create the .avi and .csv files (is the same name to the both). The line 264 allows create and save the .avi file **Plase make sure to complet a valid path to save the file !**

```python
   262        # Is saved the .avi and .csv files names.
   263        self.FileName = str(self.Text_File_Name.text())
   264        VideoFilePath =  str("...\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ " + self.FileName + ".avi")
```

* The line 492 allows get back the name of the animation digited by the user to name the motion sequence created. The line 493 allows create and save the .csv file **Plase make sure to complet a valid path (the same used to the .avi file) to save the file !**
 
```python
   491        # Is saved the name of the motion sequence.
   492        AnimationName = str(self.Text_Animation.text())
   493        FileName = str("...\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ " + self.FileName + ".csv")
```

* The lines 578-606 are used to load the .avi and .csv files and then be reproduced. **Please make sure that the path in the lines 580, 584 and 591 be the same and correct** this lines are used to load the .avi and .csv files created by the user. **The line 598 is used to load the motion sequences created by the GAN model, please make sure that the path that contain this files is correct !**

```python
   577        # Is loaded the .avi and .csv files.
   578        try:
   579           # .csv file. 
   580           FileName = str("...\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ " + str(self.Text_File_Name.text()) + ".csv")
   581           File = pd.read_csv(FileName, header = 0)
   582            
   583            # .av file.
   584            self.VideoName = str("...\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ " + str(self.Text_File_Name.text()) + ".avi")
   585            self.Video = cv2.VideoCapture(self.VideoName)
   586          
   587            PlayVideo = 1                                                       # Check the correct load.    
   588        except:
   589           try:
   590                # .csv file. 
   591                FileName = str("...\DataBaseCreatorSecuenceOfMovements\BaseDeDatos\ " + str(self.Text_File_Name.text()) + ".csv")
   592                File = pd.read_csv(FileName, header = 0)
   593                
   594                PlayVideo = 2                                                   # Check the partial load.
   595           except:
   596                try:
   597                   # .csv file 
   598                   FileName = str("...\DataBaseCreatorSecuenceOfMovements\DataBaseGeneratedByRNA\ " + str(self.Text_File_Name.text()) + ".csv")
   599                   File = pd.read_csv(FileName, header = None)
   600                    
   601                   PlayVideo = 2                                                # Check the partial load.
   602                except:
   603                   PlayVideo = 0                                                # Check the incorrect load.
   604                    pass
   605                pass
   606           pass
```

### Code explanation:

The system works using the GUI developed to perfom its two functions, the first one create new motion sequences using the kinect camera to catch all the moves of the user and then send it to be reproduce by the robot. The second function allows load the animations created and make that the robot reproduce it.

When the system start, the GUI is thrown and the system keep working until this is closed. The first step to use the system is do the connection with the robot using a Ip and Port number. If the connection is succesful the user can connect the kinect camera to create a new motion sequence. To do that, the user must to digit the name which will be saved the .avi and .csv files, and then the user can start to record the motion sequences, after of few seconds the recording ended and the user has the option to send the motion sequence to the robot reproduce it, or restart the recording. After that the motion sequence was reproduce by the robot, the user can save the motion sequence digiting the name of the animantion and then saving it. Also, the user has the option to load an animation saved previously, or an animation created by the GAN model digiting the name of the animation and loading it.

* The image shows the general flowchart of the process.
![flowchart animation data base creator general process](https://user-images.githubusercontent.com/31509775/33232419-2927da68-d1d4-11e7-847f-1ddd8e01277e.PNG)

***Record a new motion sequence***

The system get back the name of the .avi and .csv file digited by the user, the lists used to save the angles orientation of the body joints are cleared, and finally start the recordign, saving the image gotten by the kinect and computing the skeleton tracking to come back the eulerian angles orientation while five seconds.

* The image shoes the flowchart to the record a motion sequence process.
![flowchart animation data base create new motion sequence](https://user-images.githubusercontent.com/31509775/33232501-87be3cce-d1d5-11e7-8c68-26f83bf0676b.PNG)

The ***skeleton tracking*** process start to save, and then show in the GUI, the new color frame catching by the Kinect and searching for a body in it. The kinect can recognize until six bodies at the same time, so just the information of the nearest body is used. The kinect can compute the spatial points and the orientation of each bodi joint, so the next step is come back and save this information from the kinect and draw the skeleton representation on the user body in the image.

* The image shows the flowchart of the skeleton tracking process.
![flowchart animation data base creator skeleton tracking process](https://user-images.githubusercontent.com/31509775/33232543-2a570f4c-d1d6-11e7-9373-4cb949525136.PNG)

To show image gotten by the Kinect to the user using the GUI, is necessary create a surface to "paste" the image in it, and then create a RGB image from the surface. Is necessary modify the structure of the matrix resulting to generate a 3-D matrix with the RGB format and the size of the GUI frame. Finally must be created a pixel map from the RGB image to be "printed" in the GUI frame.

* The image shows the flowchart of the show image process.
![flowchart animation data base createor draw image](https://user-images.githubusercontent.com/31509775/33232551-4c4703f0-d1d6-11e7-9b18-984408f6cd0e.PNG)

To draw the skeleton representation on the user's body in the image, is necessary use the spatial coordinates of two adjacent joints, for example, to draw the head is necesary use the head and the neck spatial coordinates, to draw the right arm is necessary use the right shoulder and right elbow spatial coordinates. 

* The image shows the logical process to draw the skeleton tracking.
![flowchart animation data base creator draw body](https://user-images.githubusercontent.com/31509775/33232566-74e11684-d1d6-11e7-9bd8-18c4b2e0bf40.PNG)

The process to draw the "bones"of the body is a simple process that verify if each joint have been tracked correctly or not, and use the spatial coordinates (x, y) from the first joint to start the bone and the coordiantes of the second joint to end the bone. Is the both joints were tracked correctly the bone will be drawn with a green color, if one of them was not tracked correctly the bone will be red.

* The image shows the flowchart to the process to draw the bones.
![flowchart animation data base creator draw bones](https://user-images.githubusercontent.com/31509775/33232575-930be350-d1d6-11e7-8cf6-3ae08dcfbe77.PNG)

To get the eulerian angles (Yaw, Roll, Pitch) is necessary verify if the respective joint have been tracked correctly, if this the case, is get back the quaternion that contain the orientation and then is calculate the eulerian angles from it; is the joint was not tracked correctly its eulerian angles are saved with the "none" value. Are saved 16 angles, the process begin with the computing of the waist angles to guarantee tha have been done the skeleton tracking, and then are computed all the necessaries angles of each joint following a logic process.

* The image shows the flowchart to the computing eulerian angles process.
![flowchart animation data base creator get eulerian angles process](https://user-images.githubusercontent.com/31509775/33232586-d3922a10-d1d6-11e7-8d90-f8e26f40c44a.PNG)

To ***send to Pepper the motion sequence*** first is ordered to the robot put its joints in the initial position, then the angles are verify to delete any "none" value, if is found a none value this is replaced by the previous correct value. The control of the body start sending the waist angles values, and then the head angles values, next is sent the right arm angles values and finally the left arms angles values. All angles values must to be verify before to be sent, because the robot has a limitations in its joints moves ranges.

* The image shows the flowchart to send the angles values to the robot Pepper joints.
![flowchart animation data base creator send to pepper motion sequence process](https://user-images.githubusercontent.com/31509775/33232735-1a7ece54-d1d9-11e7-9e11-2753260ce3be.PNG)

The process to verify the angles values before send it to the robot, is based in the restrictions of it (you can go furthere finding more information about Pepper restrictions of moves [in this documentation](http://doc.aldebaran.com/2-5/family/pepper_technical/joints_pep.html)). Each angle is verify to know is belongs to a specifig range, depending of the position to the adjacent joints, the value of a specific joints can has a different move range. If the value is within the range limits,  is sent, if not, is aproximated to the nearest limit and then is sent.

* The image shows the flowchart to verify the angles values process.
![flowchart animation data base creator verify angles values](https://user-images.githubusercontent.com/31509775/33232745-49f1332a-d1d9-11e7-83af-993f7f8c90f6.PNG)

Finally the angles are saved in the .csv file created.

***Load and reproduce an animation***

To load an animation created and saved previously, the system come back the name of the .avi and .csv files digited by the user, and clear the lists used to have the angles information to each joint. The files are loaded and the angles information is saved in the corresponding lists, if the .avi file was loaded succesfull, the video and the motion sequences reproduced by the robot are playing at the same time. 

* The image shows the flowchart of the load animation process.
![flowchart animation data base creator load animation process](https://user-images.githubusercontent.com/31509775/33232795-52e7021a-d1da-11e7-92ce-a5b40300bac8.PNG)

### System user guide:

* The image shows the GUI of the system.
![animation data base creator gui](https://user-images.githubusercontent.com/31509775/33233449-f6d652dc-d1e3-11e7-99a6-88646572f664.PNG)

When the system is started the GUI appear. The first step to use the system is do the connection with the robot Pepper, or the virtual robot that you can use with the Choregraphe software. You need the Ip number (127.0.0.1 in the mayorie of the cases) and the Port number (you can find the port number of the virtual robot in the Choregraphe suit going to Edit -> Preferences -> Virtual Robot). Now the user can create a new animation or load an animation created previouly.

* The image shows how connect the system with the virtual robot.
![ip and port number gif](https://user-images.githubusercontent.com/31509775/33233811-3780e594-d1ea-11e7-96fc-1c5650f205aa.gif)

To create a new animation, the user has to connect the kinect and digit the name which he wants save the .avi and .csv file. Now the user can start to record the motion sequence and when the recording ends, the user can send the motion sequence to be reproduce by the robot. After that, if the user wants, he can save the motion sequences, digiting the name of the animation and save the data.

* The image shows an example to create of a new motion sequence.
![create new animation gif](https://user-images.githubusercontent.com/31509775/33233626-d65051cc-d1e6-11e7-92b1-40604bb48c2c.gif)

To load an animation saved previously, the user must to digit the name of the .avi and .csv files, and load the data. The system goes to reproduce the video (if this exists) and the motion sequence.

* The image shows an example to load an animation.
![load an animation gif](https://user-images.githubusercontent.com/31509775/33233735-9cb3c780-d1e8-11e7-9998-5aefd9d91463.gif)

**Click on the images to see them with a better quality**


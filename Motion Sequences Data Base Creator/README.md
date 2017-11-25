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
   601                    PlayVideo = 2                                                # Check the partial load.
   602                except:
   603                   PlayVideo = 0                                                # Check the incorrect load.
   604                    pass
   605                pass
   606           pass
```

### Code explanation:





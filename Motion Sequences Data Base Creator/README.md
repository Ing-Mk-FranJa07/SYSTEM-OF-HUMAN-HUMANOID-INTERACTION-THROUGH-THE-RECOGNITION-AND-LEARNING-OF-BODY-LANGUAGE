# Animation (motion sequences) data base creator.

### Description:

The script developed [AnimationDataBaseCreator.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Motion%20Sequences%20Data%20Base%20Creator/AnimationDataBaseCreator.py) works using the GUI created **AnimationDataBaseCreatorGUI.ui** to allow to the user created motion sequences used to be reproduce by the [Robot Pepper](https://www.ald.softbankrobotics.com/en/robots/pepper). This sequences will be used to the robot interact with the persons performing a coherent behavior with the mood of the persons and the conversation that is having with them, using the system [Recogniton and learning body lenguage system](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Complet%20Project), also the data base created with this tool can be used to train the [Genarative Adversarial Network (GAN) model built](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Create%20Motion%20Sequences) to increase the data base of the animations and give to the robot original animations developed by the network.

The system works usign the **Microsoft Kinect V2 Camera** to get the skeleton tracking and compute the angles orientation in the format (Yaw, Roll, Pitch) of each joint; using this system the user can perfom a motion sequence with his body and then this sequence goes to be reproduce by the robot, "teaching" to the robot how behave in different situations. The user can get the feedback of his moves in the image showed in the GUI. The animation can ve saved in a .csv file after have been reproduce by the robot. The system create a .avi file with the images of the user performing the motion sequence.

With the system the user can load and reproduce the video and the animations saved previously, also it can be reproduce the motion sequences created by the GAN model.

* The images shows a graphical representation of the system described.
![animation data base creator system](https://user-images.githubusercontent.com/31509775/33232003-ea41868e-d1cc-11e7-810d-f682780b3b6a.PNG)


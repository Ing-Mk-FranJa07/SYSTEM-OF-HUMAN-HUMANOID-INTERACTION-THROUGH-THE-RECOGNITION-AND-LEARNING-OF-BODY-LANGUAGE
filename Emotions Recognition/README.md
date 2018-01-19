# Recognition of emotions and conversation tool.

### Description:

The script [RecognitionOfEmotions.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/RecognitionOfEmotions.py) use the GUI **RecognitionOfEmotionsGUI.ui** developed to show the perform of the [Neural network model](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) built to recognize the mood of one person and classify it using six categories: Happy, Sad, Angry, Surprised, Reflexive and Normal. 

The system use the **Microsoft Kinect V2 Camera** to get the skeleton tracking and compute the angles orientation in the format (Yaw, Roll, Pitch) of each joint; the GUI shows the image gotten with the Kinect and the tracking. The neural network model use the joints orientation to determined the emotion that the user is represented with his body posture. The emotion is showed to the user using emojies in the GUI.

This system also has an option that allow have a conversation with a chatbox, this is possible doing speech recognition, to transform the user's speech into text string and then implementing a [Long Short Term Memory network (LSTM)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) which is a special kind of [Recurrent neural networks (RNN)](http://www.felixgers.de/papers/phd.pdf), capable of learning long-term dependencies. 

* The image is a graphical representation of the system described.
![recognition of emotions graphic](https://user-images.githubusercontent.com/31509775/32928110-61e7ab6a-cb1e-11e7-990b-3b2147f6dc75.PNG)

### RNA models used:

The ANN built to perform the recognition of the emotions using the body posture data, is a two hidden layers network that has 23 inputs and 6 ouputs that generate a codification to each emotion class designated.

* The image is a representation of the neural network built.
![ann classification problem](https://user-images.githubusercontent.com/31509775/32928269-202f9f10-cb1f-11e7-8cfe-25d3a82e2511.PNG)

The LSTM were introduced by [Hochreiter & Schmidhuber (1997)](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory), the LSTM, like all RNN, have the form of a chain of repeating modules of a neural network; but the difference with the regulars RNN is that this newtworks have reapeting modules with a very simple structure (a single tanh layer); and the LSTM has a reapting module with a different structure, in fact there are four neural network layers interacting in a very special way.

* The image illustrates the repeating module of a normal RNN and the LSTM network.
![lstm repeating module](https://user-images.githubusercontent.com/31509775/32632538-ffd338f6-c571-11e7-91e5-ac63bf978448.png)

The LSTM model used to introduce the conversation option was inspired in the [Neural conversation model](https://arxiv.org/pdf/1506.05869v3.pdf) which is caracterize by use one sequence to get another sequence, seq2seq, this model was built using the [Chatterbot wrapper](http://chatterbot.readthedocs.io/en/stable/index.html), disponible to python, which is a library that makes easier generate automates responses to string inputs using a selectrion of machine learning algorithms, including RNN and LSTM models, to produce different types of responses. 

* The image represents the LSTM model implemented with the Chatterbot tool.
![lstm](https://user-images.githubusercontent.com/31509775/32928662-1654f376-cb21-11e7-9464-0a1ebd1df5c9.png)

### Hardware and software requirements:

* Microsoft Kinect V2 sensor.

![kinect](https://user-images.githubusercontent.com/31509775/32930198-222ed504-cb2b-11e7-8455-ba7d30df2631.jpg)

* Microsoft Kinect V2 adapter. 

![kinect adapter](https://user-images.githubusercontent.com/31509775/32930206-2a22a600-cb2b-11e7-86f9-96ecb8669ddc.jpg)

* Optional: GPU card with CUDA Compute Capability 3.0 or higher [(List of supported GPU cards in NVIDIA Documentation)](https://developer.nvidia.com/cuda-gpus).

![gpu](https://user-images.githubusercontent.com/31509775/32930230-5831bcfc-cb2b-11e7-8005-4cac20045a18.png)

This system was developed using **PYTHON 2.7 (32 bits) in WINDOWS 10** to run correctly this script is necessary first to have installed:

Is recommended install [Anaconda (Python 2.7 (32 bits) version](https://www.anaconda.com/download/#windows) to get easier the packages necessaries. 

* [Kinect for Windows SDK version 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561): Microsoft website link with the installation instructions.
* [pykinect2](https://github.com/Kinect/PyKinect2): GitHub link, this repository has all the instructions to use the Kinect V2 with Python.
* [PyQT4 GPL version 4.11.4 for Python 2.7 (32 bits)](https://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe/download): Direct downlad link.
* [pygame version 1.9.2](http://www.pygame.org/news): pygame website link, you can found the download option for the pygame version 1.9.2 there.
* [keras version 2.0.6](https://keras.io/#installation): keras website link with all installation instructions.
* [Theano version 0.9.0 (keras backend engine)](http://deeplearning.net/software/theano/install_windows.html): theano windows installation instructions link.
* [speech recognition version 3.7.1](https://pypi.python.org/pypi/SpeechRecognition/): pypi.python website, all the installation instruction are specified there. (possible requirements: pyaudio version 0.2.11).
* [chatterbot version 0.7.6](http://chatterbot.readthedocs.io/en/stable/setup.html): chatterbot website installation instructions link.
* [cv2 version 3.0.0](https://docs.opencv.org/3.3.1/d5/de5/tutorial_py_setup_in_windows.html): OpenCV website link.
* numpy version 1.12.1.
* ctypes version 1.1.0

Optional software:

* [CUDA® Toolkit 8.0](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/): Ensure that you append the relevant Cuda pathnames to the %PATH% environment variable as described in the NVIDIA documentation.
* The NVIDIA drivers associated with CUDA Toolkit 8.0.
* [cuDNN v6.1](https://developer.nvidia.com/cudnn): Note that cuDNN is typically installed in a different location from the other CUDA DLLs. Ensure that you add the directory where you installed the cuDNN DLL to your %PATH% environment variable.

### WARNINGS:

* To create a new corpus data in the chatterbot library, is necessary go to the chatterbot_corpus folder and then into the data folder (installed if you use Anaconda in: **C:\...\Anaconda2_Win32\Lib\site-packages\chatterbot_corpus\data**) and create a new folder with the name "Pepper_Speech" (this was the name used by the author to create the folder in which was saved the corpus with the dialogues of the Conversation with Pepper) **Is important that if you select a different name, change the corpus name selector in the line 62 of the code**), and then create a file with the name **"myown.yml"** (you can open this file with any text editor) and paste the dialogues: [Conversation with Pepper.txt](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/Conversation%20with%20Pepper.txt). 

```python
    54  # Is created a model to generate a chatbox.
    55  PepperSay = ChatBot('Pepper Answers', 
    56                       logic_adapter = ["chatterbot.logic.MathematicalEvualation",
    57                                        "chatterbot.logic.TimeLogicAdapter",
    58                                        "chatterbot.logic.BestMatch"])
    59
    60  # Is trained the chatbox model.
    61  PepperSay.set_trainer(ChatterBotCorpusTrainer)
    62  PepperSay.train("chatterbot.corpus.Pepper_Speech")
```

* The line 62 allow the training of the chatterbot model, when the training end, the file **"db.sqlite3"** is created, if you want, you can comment this line to avoid the time of the training, but please make shure that you have downloaded the **"db.sqlite3"** file into the folder with all scripts presented in this repository.

* The line 69 allow load the ANN model built (you can re-train or modify this model with the script [RNA_Emotions_BodyPosture_Keras_Tensorflow.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions)), the model is avaible in this repository with the name **"Model_RNA_Recognition_Of_Emotions"**. **Please make sure that the path of the file is correct !**

```python
    68  # Is loaded the neural network model.
    69  RNA = load_model('...\Model_RNA_Recognition_Of_Emotions')
```

* The line 108 allow load the GUI developed to use the system. **Please make sure that the path of the file is correct !**

```python
    107        # Is loaded the file that get the GUI.       
    108       self.MyGUI = uic.loadUi('...\RecognitionOfEmotionsGUI.ui', self)
```

### Code explanation:

The perform of the system is based in two big function that are integrated in the GUI developed. When the system is started, the GUI is thrown and it keep working until the user close the GUI. The user can select select just one of the functions at the same time, the first one connect the Kinect V2 camera and start the process to recognize the emotion of the user using the posture of his body, this function can be activated or deactivated in any momment. The second function disconnect the Kinect and it acces to the microphone to get the speec of the user and generate an answer. The representation of the emotion recognized and the answer generated by the system are showed to the user in the GUI.

* The image show the general flowchart of the system.
![recognition of emotions general flowchart](https://user-images.githubusercontent.com/31509775/32955129-1384272c-cb83-11e7-93c8-78f8b97cfba4.PNG)

***Skeleton tracking and recognition of emotions function***

The skeleton tracking process allow get back the image gotten by the Kinect, and also, the body information about the joints tracked from the nearest body to the Kinect detected, the spatial ubication of each joint and their orientation. In this process the image es showed to the user and is drawn the skeleton representation on the user body in the image. The final objective of this process is to save the eulerian angles orientation of each joint. 

* The image shows the general flowchart of the skeleton tracking process.
![flowchart skeleton tracking general process](https://user-images.githubusercontent.com/31509775/32955038-c7d643b4-cb82-11e7-91e7-4baee5968b90.PNG)

To show the image gotten by the Kinect to the user using the GUI, is necessary create a surface to "paste" the image in it, and then create a RGB image from the surface. Is necessary modify the structure of the matrix resulting to generate a 3-D matrix with the RGB format and the size of the GUI frame. Finally must be created a pixel map from the RGB image to be "printed" in the GUI frame.

* The image shows the flowchart of the process to show the image gotten by the Kinect in the GUI.
![flowchart show image](https://user-images.githubusercontent.com/31509775/32955478-1797234a-cb84-11e7-83d5-d6482d78427c.PNG)

To draw the skeleton representation on the user's body in the image, is necessary use the spatial coordinates of two adjacent joints, for example, to draw the head is necesary use the head and the neck spatial coordinates, to draw the right arm is necessary use the right shoulder and right elbow spatial coordinates. 

* The image shows the logical process to draw the skeleton tracking.
![flowchart draw body](https://user-images.githubusercontent.com/31509775/32955651-9d23ae84-cb84-11e7-91b7-1035102d79a4.PNG)

Draw each "bone" of the body is a simple process that verify if each joint have been tracked correctly or not, and use the spatial coordinates (x, y) from the first joint to start the bone and the coordiantes of the second joint to end the bone. Is the both joints were tracked correctly the bone will be drawn with a green color, if one of them was not tracked correctly the bone will be red.

* The image shows the flowchart to the process to draw the bones.
![flowchart draw bones](https://user-images.githubusercontent.com/31509775/32955857-29cdbd5c-cb85-11e7-830a-82ebf09194be.PNG)

To get the eulerian angles (Yaw, Roll, Pitch) is necessary verify if the respective joint have been tracked correctly, if this the case, is get back the quaternion that contain the orientation and then is calculate the eulerian angles from it; is the joint was not tracked correctly its eulerian angles are saved with the "none" value. At the end is conformed an array that contains 23 angles. The process begin with the computing of the waist angles to guarantee tha have been done the skeleton tracking, and then are computed all the necessaries angles of each joint following a logic process.

* The image shows the flowchart to the computing eulerian angles process.
![flowchart computing body joints orientation to recognize emotions](https://user-images.githubusercontent.com/31509775/32956595-2d7c5272-cb87-11e7-90db-745f5508ce87.PNG)

Finally, the emotion is recognize using the neural network model built and load previously which input is a 23 position array and output is a binary array of 6 positions, to generate a one hot codification to each emotion class. Depending of the codification is showed an emojie to represent the emotion in th GUI.

* The image shows the flowchart of the process that allow recognize the emotion.
![flowchart recognition of emotion process](https://user-images.githubusercontent.com/31509775/33040309-a7e2d760-ce08-11e7-9ce2-75e8ea124e91.PNG)

***Speech recognition and generate answers function***

The second function of the system allow generate an answer using a LSTM neural model built with the chatterbot library, from the speech of the user that is gotten with the microphone of the pc (you can use a external microphone to) and then is tranformated into a text string that will be the input to the LSTM neural model. The text string and the answer generated are showed in the GUI.

* The image shows the flowchart of the process that allow the speech recognition and the answers generation.
![flowchart speech recognition](https://user-images.githubusercontent.com/31509775/35164998-a0d487d4-fd1a-11e7-99f1-70c3342fc22c.PNG)

### System user guide.

When the system is started, the GUI appear. The user can push on the button with the text "Connect Kinect" to start the Kinect camera and use the emotion recognition function. Also, when the button is pushed, the text change to "Disconnect Kinect", pushing again end the recognition of emotions function.

* The image shows the GUI highlighting the button mentioned.
![recognition of emotions connect kinect](https://user-images.githubusercontent.com/31509775/32958701-44911f5a-cb8d-11e7-9e7c-5aeb7a462b50.png)

This function works with the neural model trained by the author, in which the user can be sit or stand up. The advice is that you create your own data base with different persons trying to represent the emotions with theirs bodys to re-train the neural model and increase the performance of the system.

* The image shows the recognize of emotions performing a few positions being stand up.
![emotions-recognition-stand-up-gif](https://user-images.githubusercontent.com/31509775/32961271-ec671d80-cb95-11e7-81c9-8508b1f0d820.gif)

* The image shows the recognize of emotions performing a few positions being sit.
![emotions-recognition-sit-down-gif](https://user-images.githubusercontent.com/31509775/32960893-9c1d4f6c-cb94-11e7-954b-d8808c3bda19.gif)

In any moment the user can activate the speech recognition function, clicking on the checkbox "Speech recognitions deactivated", when this functions is activated the Kinect V2 camera is disconnect, this happen because the actualization of the GUI take more time that the time that need the code to be ran, so the GUI can´t show the new image, the change in the emojies and the speech and answer generated at the same time. Click again on the checkbox "Speech recognition activated" to deactivate the speech recognition function and enable again the emotion recognition.

* The image shows the GUI highlighting the checkbox mentioned.
![recognition of emotions speech recognition](https://user-images.githubusercontent.com/31509775/32961664-70ac0eb0-cb97-11e7-95ce-b1ff4b253ca6.png)

The speech recognition take a time while transform the speech to text. The GUI don´t show the informative messages, so if you can open a console to follow the changes it will do better the perform of the function. Also, is recommended have a good microphone and be in a low noise envairoment. The chatterbot model can generate answer to similar sentences to that were trained, is not necessary that be the same sentences. 

* The image shows a few sentences spoken by the user and the answer generated.
![speech recognition test gif](https://user-images.githubusercontent.com/31509775/32961706-92d650b8-cb97-11e7-9fb2-22ad5ed9a523.gif)

**Click on the images to see them with better quality** 


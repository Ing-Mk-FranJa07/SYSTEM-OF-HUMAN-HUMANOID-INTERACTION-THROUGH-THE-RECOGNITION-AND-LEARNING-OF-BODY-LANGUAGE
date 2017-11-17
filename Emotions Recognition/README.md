# Recognition of emotions and conversation tool.

### Description:

The script [RecognitionOfEmotions.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/RecognitionOfEmotions.py) use the GUI **RecognitionOfEmotionsGUI.ui** developed to show the perform of the [Neural network model](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) built to recognize the mood of one person and classify it using six categories: Happy, Sad, Angry, Surprised, Reflexive and Normal. 

The system use the Kinect V2 camera to get the skeleton tracking and compute the angles orientation in the format [(Yaw, Roll, Pitch)] of each joint; the GUI shows the image gotten with the Kinect and the tracking. The neural network model use the joints orientation to determined the emotion that the user is represented with his body posture. The emotion is showed to the user using emojies in the GUI.

This system also has an option that allow have a conversation with a chatbox, this is possible doing speech recognition, to transform the user's speech into text and then implementing a [Long Short Term Memory network (LSTM)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) which is a special kind of [Recurrent neural networks (RNN)](http://www.felixgers.de/papers/phd.pdf), capable of learning long-term dependencies. 

* The image is a graphical representation of the system described.
![recognition of emotions graphic](https://user-images.githubusercontent.com/31509775/32928110-61e7ab6a-cb1e-11e7-990b-3b2147f6dc75.PNG)

### RNA models used:

[The ANN built to perform the recognition of the emotions](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) using the body posture data, is a two hidden layers network that has 23 inputs and 6 ouputs that generate a codification to each emotion class designated.

* The image is a representation of the neural network built.
![ann classification problem](https://user-images.githubusercontent.com/31509775/32928269-202f9f10-cb1f-11e7-8cfe-25d3a82e2511.PNG)

The LSTM were introduced by [Hochreiter & Schmidhuber (1997)](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory), the LSTM, like all RNN, have the form of a chain of repeating modules of a neural network; but the difference with the regulars RNN is that this newtworks have reapeting modules with a very simple structure (a single tanh layer); and the LSTM has a reapting module with a different structure, in fact there are four neural network layers interacting in a very special way.

* The image illustrate the repeating module of a normal RNN and the LSTM network.
![lstm repeating module](https://user-images.githubusercontent.com/31509775/32632538-ffd338f6-c571-11e7-91e5-ac63bf978448.png)

The LSTM model used to introduce the conversation option was inspired in the [Neural conversation model](https://arxiv.org/pdf/1506.05869v3.pdf) which is caracterize by use one sequence to get another sequence, seq2seq, this model was built using the [Chatterbot wrapper](http://chatterbot.readthedocs.io/en/stable/index.html), disponible to python, which is a library that makes easier generate automates responses to string inputs using a selectrion of machine learning algorithms, including RNN and LSTM models, to produce different types of responses. 

* The image represent the LSTM model implemented with the Chatterbot tool.
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

* To create a new corpus data in the chatterbot library, is necessary go to the chatterbot_corpus folder and then into the data folder (installed if you use Anaconda in: **C:\...\Anaconda2_Win32\Lib\site-packages\chatterbot_corpus\data**) and create a new folder with the name "Pepper_Speech" (this was the name used by the author to create the folder in which was saved the corpus with the dialogues of the [Conversation with Pepper.txt](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/Conversation%20with%20Pepper.txt) **Is important that if you select a different name, change the corpus name selector in the line 62 of the code**), and then create a file with the name **"myown.yml"** (you can open this file with any text editor) and paste the dialogues: [Conversation with Pepper.txt](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/Conversation%20with%20Pepper.txt). 
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

* The line 69 allow load and save the ANN model built (you can re-train or modify this model with the script [RNA_Emotions_BodyPosture_Keras_Tensorflow.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions)), the model is avaible in this repository with the name **"Model_RNA_Recognition_Of_Emotions"**. **Please make shure that the path of the file is correct !**
```python
    68  # Is loaded the neural network model.
    69  RNA = load_model('...\Model_RNA_Recognition_Of_Emotions')
```
### Code explanation:


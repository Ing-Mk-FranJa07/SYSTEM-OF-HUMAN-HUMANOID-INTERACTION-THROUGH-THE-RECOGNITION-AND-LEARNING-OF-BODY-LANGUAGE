# Recognition of emotions and conversation tool.

### Description:

* The image is a graphical representation of the system described.
![recognition of emotions graphic](https://user-images.githubusercontent.com/31509775/32928110-61e7ab6a-cb1e-11e7-990b-3b2147f6dc75.PNG)


The script [RecognitionOfEmotions.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/RecognitionOfEmotions.py) use the GUI **RecognitionOfEmotionsGUI.ui** developed to show the perform of the [Neural network model](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) built to recognize the mood of one person and classify it using six categories: Happy, Sad, Angry, Surprised, Reflexive and Normal. 

The system use the Kinect V2 camera to get the skeleton tracking and compute the angles orientation in the format [(Yaw, Roll, Pitch)] of each joint; the GUI shows the image gotten with the Kinect and the tracking. The neural network model use the joints orientation to determined the emotion that the user is represented with his body posture. The emotion is showed to the user using emojies in the GUI.

This system also has an option that allow have a conversation with a chatbox, this is possible doing speech recognition, to transform the user's speech into text and then implementing a [Long Short Term Memory network (LSTM)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) which is a special kind of [Recurrent neural networks (RNN)](http://www.felixgers.de/papers/phd.pdf), capable of learning long-term dependencies. 


### RNA models used:

[The ANN built to perform the recognition of the emotions](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) using the body posture data, is a two hidden layers network that has 23 inputs and 6 ouputs that generate a codification to each emotion class designated.

* The image is a representation of the neural network built.
![ann classification problem](https://user-images.githubusercontent.com/31509775/32928269-202f9f10-cb1f-11e7-8cfe-25d3a82e2511.PNG)

The LSTM were introduced by [Hochreiter & Schmidhuber (1997)](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory), the LSTM, like all RNN, have the form of a chain of repeating modules of a neural network; but the difference with the regulars RNN is that this newtworks have reapeting modules with a very simple structure (a single tanh layer); and the LSTM has a reapting module with a different structure, in fact there are four neural network layers interacting in a very special way.

* The image illustrate the repeating module of a normal RNN and the LSTM network.
![lstm repeating module](https://user-images.githubusercontent.com/31509775/32632538-ffd338f6-c571-11e7-91e5-ac63bf978448.png)

The LSTM model used to introduce the conversation option was inspired in the [Neural conversation model](https://arxiv.org/pdf/1506.05869v3.pdf) which is caracterize by use one sequence to get another sequence, seq2seq, this model was built using the [Chaterbot wrapper](http://chatterbot.readthedocs.io/en/stable/index.html), disponible to python, which is a library that makes easier generate automates responses to string inputs using a selectrion of machine learning algorithms, including RNN and LSTM models, to produce different types of responses. 

* The image represent the LSTM model implemented with the Chaterbot tool.
![lstm](https://user-images.githubusercontent.com/31509775/32928662-1654f376-cb21-11e7-9464-0a1ebd1df5c9.png)

### Hardware and software requirements:

* Microsoft Kinect V2 sensor.
![kinect](https://user-images.githubusercontent.com/31509775/32930198-222ed504-cb2b-11e7-8455-ba7d30df2631.jpg)

* Microsoft Kinect V2 adapter. 
![kinect adapter](https://user-images.githubusercontent.com/31509775/32930206-2a22a600-cb2b-11e7-86f9-96ecb8669ddc.jpg)

* Optional: Nvidia GPU.
![gpu](https://user-images.githubusercontent.com/31509775/32930230-5831bcfc-cb2b-11e7-8005-4cac20045a18.png)


This system was developed using **PYTHON 2.7 (32 bits) in WINDOWS 10** to run correctly this script is necessary first to have installed:




# Generative Adversarial Network (GAN): Neural network model used to increase the motion sequences data base created to perform the interaction between the Robot Pepper and the humans.

### Description:

The [Genrative Adversarial Network](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) is a recent development in deep learning introduced by Ian Goodfellow en 2014. This model attend the problem of unsupervised learning by training two deep networks called Generator and Discriminator. The both networks compete and cooperate with the other to learn how to perform their tasks.

The GAN model has been explained in the most of the times like a case of a counterfeiter (Generative network) and a policeman (Discriminator network), initially the counterfeiter create and show to the policeman a "fake money" (the data); the policeman says that it's face and gives feedback to the counterfeiter why the money is fake. Now the counterfeiter trys to make new fake money based on the feedback that it received, and show to the policeman the fake money again. Policeman decide if the money is or not fake and offers a new feedback to counterfeiter. This cycle continues indefinitely while the counterfeiter goes tcreating each time a better fake money and it will be looking so similar to the real money, finally the policeman is fooled.

Teoretically the GAN is very simple, but build a model that works correctly is very difficult, because there are two deep networks coupled together making back propagation of gradients twice as challenging. Exist an interesting example of the application to the GAN models using [Deep Convolutional GAN (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf%C3%AF%C2%BC%E2%80%B0) that demonstrated how to build a practical GAN to learn by itselvef how to synthesize new images.

In this case, the script [RNA_Generate_Animations_GAN.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Nueral%20Networks/Create%20Motion%20Sequences/RNA_Generate_Animations_GAN.py) build a GAN model used to try to create new motion sequences, and not images, very similar to the original motion sequences that was created to be played to the [Robot Pepper](file:///C:/Program%20Files%20(x86)/Aldebaran/Choregraphe%20Suite%202.5/share/doc/home_pepper.html). Was decided try to build a GAN model, because the structure of the data is a matrix similar to the structure of the a gray image.

The image show a representation of the GAN model created.

![gan model](https://user-images.githubusercontent.com/31509775/32291493-3500cbd2-bf0b-11e7-8102-68e7da40dbc6.PNG)

This script was developed using **PYHTON 3.6 (64 bits) in WINDOWS 10** and the following wrappers.
* Keras version 2.0.6
* Tensorflow version 1.2.1 (keras backend engine)
* pandas version 0.19.2
* numpy version 1.11.3
* matplotlib 2.0.0

To use correctly this script, please consider the follow steps.

# First step: Create Data Base.

It's necessary to have a data base, and it's possible create it with the tool: [AnimationDataBaseCreator.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Motion%20Sequences%20Data%20Base%20Creator), this tool generate a .csv file in which is saved a matrix that containing 39 rows and 15 columns, each column has the information about a motion sequence of each [Robot Pepper joint](file:///C:/Program%20Files%20(x86)/Aldebaran/Choregraphe%20Suite%202.5/share/doc/family/pepper_technical/joints_pep.html)

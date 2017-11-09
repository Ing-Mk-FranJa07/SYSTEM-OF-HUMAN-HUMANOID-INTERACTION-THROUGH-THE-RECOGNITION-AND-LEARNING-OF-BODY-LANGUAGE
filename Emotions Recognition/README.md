# Recognition of emotions and conversation tool.

### Description.

The script [RecognitionOfEmotions.py](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/blob/master/Emotions%20Recognition/RecognitionOfEmotions.py) use the GUI **RecognitionOfEmotionsGUI.ui** developed to show the perform of the [Neural network model](https://github.com/Ing-Mk-FranJa07/SYSTEM-OF-HUMAN-HUMANID-INTERACTION-THROUGH-THE-RECOGNITION-AND-LEARNING-OF-BODY-LANGUAGE/tree/master/Nueral%20Networks/Classify%20emotions) built to recognize the mood of one person and classify it using six categories: Happe, Sad, Angry, Surprised, Reflexive and Normal. 

The system use the Kinect V2 camera to get the skeleton tracking and compute the angles orientation in the format [(Yaw, Roll, Pitch)] of each joint; the GUI shows the image gotten with the Kinect and the tracking. The neural network model use the joints orientation to determined the emotion that the user is represented with his body posture. The emotion is showed to the user using emojies in the GUI.


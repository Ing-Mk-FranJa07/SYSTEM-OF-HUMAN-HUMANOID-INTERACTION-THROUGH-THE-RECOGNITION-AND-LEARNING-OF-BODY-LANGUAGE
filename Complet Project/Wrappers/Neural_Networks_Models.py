'''
Created on 18/09/2017

@author: Mk Eng: Francisco Javier Gonzalez Lopez.

Script that implement trained Neural Networks to:

- Recognize emotions using body language through the information get with
  Skeleton tracking like the input to a Neural Network which solve a classification
  problem, deciding the emotions or mood, that one person is expressing. 
  
- "Create" original Motion sequence perform General Alternative Network model,
  which transform noise random numbers in an matrix which has sequences of 
  joint angles.

- Perform the speech recognition, through a special wrapper with Long-Short Term Memory
  Neural Networks pre-created to predict the "answer" to a "question" and perform a 
  "conversation".

In this version of python 2.7 (32 bit) is not available the Tensorflow backend.
Is necessary use Theano backend to implemented the RNA models created with keras.
Theano backend is not compatible with the model developed to implemented the GAN.
'''
# Are imported the wrappers that have methods to implemented RNA models, and manipulated data.
import pandas as pd
import numpy as np
from keras.models import load_model                                             # Wrapper that allow load a RNA model, trained previously. 
from chatterbot import ChatBot                                                  # Wrapper that implement LSTM to create ChatBot system.
from chatterbot.trainers import ChatterBotCorpusTrainer                         # Wrapper that allow train and load pre-trained ChatBots.

class Neural_Networks_Models(object):
    '''
    Wrapper class to perform RNA models which recognize emotions, created Motion 
    animations and allow speech recognition. 
    '''
    def __init__(self):
        # Is loaded the RNA model that perform the recognitions of emotions, using the joint angles information.
        self.RNA_Emotions = load_model('...\Data_And_RNA_Models\Model_RNA_Recognition_Of_Emotions')
        
        # Is loaded the Motion animations database.
        self.AnimationsDataBase = pd.read_csv('...\Data_And_RNA_Models\Animations Sequence.csv', header = 0)
        self.AnimationsDataBase.set_index("Animation", inplace = True)
        
        # Are loaded the GAN models, that perform the creation of Motion animation sequences.
        #self.Adversarial = load_model('...\Data_And_RNA_Models\Adversarial Model')
        #self.Discriminator = load_model('...\Data_And_RNA_Models\Discriminator Model')
        #self.Generator = load_model('...\Data_And_RNA_Models\Generator Model')
        
        # Is configured a ChatBot model that allow get answers to pre-define questions.         
        self. PepperSay = ChatBot('Pepper Answers', 
                             logic_adapter = ["chatterbot.logic.MathematicalEvualation",
                                              "chatterbot.logic.TimeLogicAdapter",
                                              "chatterbot.logic.BestMatch"])
        
        self.PepperSay.set_trainer(ChatterBotCorpusTrainer)                     # Configure the train method.
        #self.PepperSay.train("chatterbot.corpus.Pepper_Speech")                 # Train the model, with a specific dataset.
        
        # Is configured a ChatBot model that allow get the animation codes consistently with the speech.
        self.PepperAnimation = ChatBot('Pepper Speech Animations')
    
        self.PepperAnimation.set_trainer(ChatterBotCorpusTrainer)               # Configure the train method.
        #self.PepperAnimation.train("chatterbot.corpus.Pepper_Speech_Animation") # Train the model, with a specific dataset.
    
    def RecognizeEmotions(self, Sequence):
        '''
        Function that allow recognize the mood, using the body language through
        the information get with the skeleton tracking, and implemented a RNA
        which solve a classification problem.
        '''
        Output = self.RNA_Emotions.predict(Sequence)
        
        Output_RNA = [round(Output[0][I]) for I in range(Output.shape[1])]
        
        return Output_RNA
    
    def SelectAnimation(self, Index):
        '''
        Function that select a Motion animation sequence.
        '''   
        if Index == "Victory":
            return np.array(self.AnimationsDataBase.loc[Index])
        
        elif Index == "Yeah":
            return np.array(self.AnimationsDataBase.loc[Index])
        
        elif Index == "Why_sad":
            return np.array(self.AnimationsDataBase.loc[Index])
        
        elif Index == "Don't_be_sad":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Scary":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Calm_down":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "What_happened":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Whats_up":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "What_are_thinking":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Are_somebody_there":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Hello_1":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Hello_2":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Regards":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "I_am_good":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "All_good":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Excellent":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Thinking":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "I_am":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "I_am_duh":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Me_and_you":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "You":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Later":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "We":
            return np.array(self.AnimationsDataBase.loc[Index])
    
        elif Index == "Show_me":
            return np.array(self.AnimationsDataBase.loc[Index])
    
        elif Index == "Sorry":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "What!":
            return np.array(self.AnimationsDataBase.loc[Index])
    
        elif Index == "I_don't_know":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Good_luck":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Noo":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Right_Right":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Cool":
            return np.array(self.AnimationsDataBase.loc[Index])

        elif Index == "Ignore":
            return np.array(self.AnimationsDataBase.loc[Index])
    
    def GenerateAnimation(self, Animation):
        '''
        Function that return a original motion animation sequence, created using
        a GAN model. 
        '''        
        # Are get two random int to select a original animation of the DataSet.
        Iteration = np.random.randint(301); Iteration = [1 if Iteration == 0 else Iteration]; Iteration = [300 if Iteration == 301 else Iteration]
        AnimationData = np.random.randint(33); AnimationData = [1 if AnimationData == 0 else AnimationData]; AnimationData = [300 if AnimationData == 301 else AnimationData]
        
        # Is loaded a original animation of the DataSet.
        FileName = ("...\Data_And_RNA_Models\DataBaseGeneratedByRNA\ NewAnimation " + str(Iteration[0][0]) + "-" + str(AnimationData[0][0]) + ".csv")
        File = pd.read_csv(FileName, header = None)
        
        # Is created an array with the file loaded.
        OriginalAnimation = np.array(File)
                
        return OriginalAnimation
        
    def GetAnswer(self, Question):
        '''
        Function that do the prediction of the Pepper's "answer" from the user's
        "questions" using a LSTM models, also is selected the Motion animation that
        is consistently with the "answer".
        '''
        Answer = self.PepperSay.get_response(Question)                          # Get the "answer".
        AnimationCode = self.PepperAnimation.get_response(Answer)               # Get the animation code index.
        
        Animation = self.SelectAnimation(AnimationCode)                         # Get the Motion animation sequence.
        
        return Answer, Animation
        
        
        
        
        
             

import numpy as np
from keras import models
from keras import layers
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from src.preprocessing import Preprocessing


class SentimentNeuralNet():
    
    def __init__(self):
        self.preprocess = Preprocessing()
        self.tokenizer = Tokenizer()
    

                
    def train_neural_net(self, x_train, y_train, x_val, y_val):
        model = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)), 
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  
            metrics=['accuracy']
        )
        
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=1000,
            batch_size=32
        )
        
        print("Training complete.")
        
class Train(SentimentNeuralNet):
    def __init__(self):
        super().__init__()
        
    def train_dataset(self, dataset):
        match dataset:
            case 'hate-speech':
                self.preprocess.get_datasets()
                self.preprocess.clean_datasets()
                data_frame = pd.read_csv(self.preprocess.clean_dataset_paths[dataset])
        
                sequences = self.tokenizer.texts_to_sequences(data_frame['tweet'])
                x = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
                print(x.dtype)

                y = data_frame['class'].astype('int32').values
                print(y.dtype) 
                
                x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=69)

                self.train_neural_net(x_train,y_train,x_val,y_val)
                print(data_frame['tweet'].head()) 
            case _:
                print("Unkown dataset provided")
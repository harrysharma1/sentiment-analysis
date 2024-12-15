import numpy as np
from keras import models
from keras import layers
import pandas as pd
from src.preprocessing import Preprocessing


class SentimentNeuralNet():
    
    def __init__(self):
        self.preprocess = Preprocessing()
    
    def train_dataset(self, dataset):
        match dataset:
            case 'hate-speech':
                data_frame = pd.read_csv(self.preprocess.clean_dataset_paths[dataset])
                
                data_frame['tweet'] = data_frame['tweet'].apply(lambda x: ' '.join(x))
                
           
                 
            case _:
                print("Unkown dataset provided")
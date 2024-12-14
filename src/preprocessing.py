import os
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string


class Preprocessing():
    def __init__(self):
        self.url = {
            "hate-speech":"https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv",
        }
        self.raw_dataset_paths ={}
        
        try:
            datasets_path = os.path.join(os.getcwd(),'datasets')
            raw_path = os.path.join(datasets_path, 'raw')
            clean_path = os.path.join(datasets_path, 'clean')

            os.makedirs(raw_path, exist_ok=True)
            os.makedirs(clean_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {e}")
            raise
        
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words('english'))
    
    def remove_punctuation(self, text):
        return "".join([i for i in text if i not in string.punctuation])
    
    def get_datasets(self):
        for filename, url in self.url.items():
            try:
                response = requests.get(url)
                path = os.path.join(os.getcwd(), 'datasets/raw', f"{filename}.csv") 
                with open(path, 'wb') as file:
                    file.write(response.content)
                self.raw_dataset_paths[filename] = path
                print(f"Data saved to: {path}")
            except requests.HTTPError as e:
                match e.response.status_code:
                    case 404:
                        print(f"Error:\nDataset not found.\nStatus code: {e.response.status_code}.")
                    case _ :
                        print(f"Error:\nUnkown.\nStatus code: {e.response.status_code}")
            except requests.ConnectionError as e:
                print(f"Error:\nHostname was not resolvable.\nFull error: {e}")
            except FileExistsError as e:
                print(f"Error:\nFile has already been created.")
                
    def clean_datasets(self):

        def clean_hate_speech(tweet):
            tweet = str(tweet)
            
            tweet = re.sub(r'RT\s+', '', tweet)
            
            tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
            tweet = re.sub(r'\@\w+|\#w+', '', tweet)
        
            tweet = re.sub(r'&amp;|&lt;|&gt;|&quot;|&#\d+;', '', tweet)
            
            tweet = tweet.encode('ascii', 'ignore').decode('ascii')
            
            tweet = re.sub(r'["""]', '', tweet)
            
            tweet = re.sub(r'[^\w\s]', ' ', tweet)
            tweet = re.sub(r'\d+', '', tweet)
            
            tweet = tweet.lower()
            
            tweet = re.sub(r'\s+', ' ', tweet)
            tweet = tweet.strip()
            tweet = ' '.join([word for word in tweet.split() if word not in self.stop_words])
            return tweet
        
        if 'hate-speech' in self.raw_dataset_paths:
            data_frame = pd.read_csv(self.raw_dataset_paths['hate-speech'])
            print(data_frame.head())
            data_frame['clean_tweet'] = data_frame['tweet'].apply(lambda x:self.remove_punctuation(x))
            print(data_frame.head())
            
            clean_csv_path = os.path.join(os.getcwd(),'datasets/clean','hate-speech.csv')
            data_frame.to_csv(clean_csv_path,index=0)
            print (f"Cleaned data saved to: {clean_csv_path}")
        else:
            print ("Not created dataset.")

                        
                        
a = Preprocessing()
a.get_datasets()
a.clean_datasets()

            
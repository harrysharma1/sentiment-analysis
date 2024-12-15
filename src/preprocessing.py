import os
import requests
import pandas as pd
from src.preprocessing_utilities import HateSpeechUtilities, SouthParkUtilities


class Preprocessing():
    def __init__(self):
        self.url = {
            "hate-speech":"https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv",
            "south-park-season-1":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-1.csv",
            "south-park-season-2":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-2.csv",
            "south-park-season-3":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-3.csv",
            "south-park-season-4":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-4.csv",
            "south-park-season-5":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-5.csv",
            "south-park-season-6":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-6.csv",
            "south-park-season-7":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-7.csv",
            "south-park-season-8":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-8.csv",
            "south-park-season-9":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-9.csv",
            "south-park-season-10":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-10.csv",
            "south-park-season-11":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-11.csv",
            "south-park-season-12":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-12.csv",
            "south-park-season-13":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-13.csv",
            "south-park-season-14":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-14.csv",
            "south-park-season-15":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-15.csv",
            "south-park-season-16":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-16.csv",
            "south-park-season-17":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-17.csv",
            "south-park-season-18":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-18.csv",
            "south-park-season-19":"https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-19.csv",
        }
        self.raw_dataset_paths = {}
        self.clean_dataset_paths = {}
        self.clean_dataset_text_col = {}
        
        try:
            datasets_path = os.path.join(os.getcwd(),'datasets')
            raw_path = os.path.join(datasets_path, 'raw')
            clean_path = os.path.join(datasets_path, 'clean')

            os.makedirs(raw_path, exist_ok=True)
            os.makedirs(clean_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {e}")
            raise
        
    def get_datasets(self):
        print("Pulling raw data from datasets...")
        for filename, url in self.url.items():
            try:
                response = requests.get(url)
                path = os.path.join(os.getcwd(), 'datasets/raw', f"{filename}.csv") 
                with open(path, 'wb') as file:
                    file.write(response.content)
                self.raw_dataset_paths[filename] = path
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
        for path in self.raw_dataset_paths.values():
            print(f"Data saved to: {path}")    
            
                
    def clean_datasets(self):
        print("Begin preprocess...")
        for filename in self.raw_dataset_paths.keys():
            match filename:
                case 'hate-speech':
                    hate_speech = HateSpeechUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['hate-speech'])
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.clean(x))
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.remove_punctuation(x))
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: x.lower())
                    data_frame['tweet'] = data_frame['tweet'].str.replace('\n',' ')
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.tokenization(x))
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.clean_tokens(x))
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.remove_stop_words(x))
                    data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'tweet'
                case 'south-park-season-1':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-1'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                case 'south-park-season-2':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-2'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-3':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-3'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                     
                case 'south-park-season-4':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-4'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                    
                case 'south-park-season-5':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-5'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                    
                case 'south-park-season-6':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-6'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                    
                case 'south-park-season-7':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-7'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-7':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-7'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                    
                case 'south-park-season-8':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-8'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                    
                case 'south-park-season-9':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-9'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'
                    
                case 'south-park-season-10':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-10'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-11':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-11'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-12':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-12'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-13':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-13'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-14':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-14'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-15':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-15'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-16':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-16'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-17':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-17'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-18':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-18'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case 'south-park-season-19':
                    south_park = SouthParkUtilities()
                    data_frame = pd.read_csv(self.raw_dataset_paths['south-park-season-19'])
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_punctuation(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
                    data_frame['Line'] = data_frame['Line'].str.replace('\n','')
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.tokenization(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.clean_tokens(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.remove_stop_words(x))
                    data_frame['Line'] = data_frame['Line'].apply(lambda x: south_park.lemmatize(x))
                    
                    clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
                    data_frame.to_csv(clean_csv_path,index=0)
                    print (f"Cleaned data saved to: {clean_csv_path}")
                    self.clean_dataset_paths[filename] = clean_csv_path
                    self.clean_dataset_text_col[filename] = 'Line'

                case _:
                    print(f"Not created clean dataset for {filename}.")
        print("Preprocess complete...") 
       

                        
                        

            
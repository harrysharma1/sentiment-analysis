import os
import requests
import pandas as pd
import kagglehub
import uuid
from src.preprocessing_utilities import HateSpeechUtilities, SouthParkUtilities, GeneralPurposeUtilitiees


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
        
    def get_all_preset_datasets(self):
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
            
                
    def clean_all_preset_datasets(self):
        print("Begin preprocess...")
        for filename in self.raw_dataset_paths.keys():
            match filename:
                case 'hate-speech':
                    hate_speech = HateSpeechUtilities()
                    self.clean_dataset_paths[filename], self.clean_dataset_text_col[filename] = hate_speech.clean(self.raw_dataset_paths[filename], filename)
                
                case 'south-park-season-1'|'south-park-season-2'|'south-park-season-3'|'south-park-season-4'|'south-park-season-5'|'south-park-season-6'|'south-park-season-7'|'south-park-season-8'|'south-park-season-9'|'south-park-season-10'|'south-park-season-11'|'south-park-season-12'|'south-park-season-13'|'south-park-season-14'|'south-park-season-15'|'south-park-season-16'|'south-park-season-17'|'south-park-season-18'|'south-park-season-19':
                    south_park = SouthParkUtilities()
                    self.clean_dataset_paths[filename], self.clean_dataset_text_col[filename] = south_park.clean(self.raw_dataset_paths[filename],filename)
                case _:
                    print(f"Not created clean dataset for {filename}.")
        print("Preprocess complete...")
       
 
    def get_dataset(self, url="https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv", filename=uuid.uuid4()):
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
            
    def clean_dataset(self, filename):
        print("Begin preprocess...")
        if filename not in self.raw_dataset_paths.keys():
            print(f"File not found make sure the that {filename}.csv can be found within the datasets/raw folder")
        else:
            match filename:
                case 'hate-speech':
                    hate_speech = HateSpeechUtilities()
                    self.clean_dataset_paths[filename], self.clean_dataset_text_col[filename] = hate_speech.clean(self.raw_dataset_paths[filename], filename)
                
                case 'south-park-season-1'|'south-park-season-2'|'south-park-season-3'|'south-park-season-4'|'south-park-season-5'|'south-park-season-6'|'south-park-season-7'|'south-park-season-8'|'south-park-season-9'|'south-park-season-10'|'south-park-season-11'|'south-park-season-12'|'south-park-season-13'|'south-park-season-14'|'south-park-season-15'|'south-park-season-16'|'south-park-season-17'|'south-park-season-18'|'south-park-season-19':
                    south_park = SouthParkUtilities()
                    self.clean_dataset_paths[filename], self.clean_dataset_text_col[filename] = south_park.clean(self.raw_dataset_paths[filename],filename)
            
                case _:
                    general_utils = GeneralPurposeUtilitiees()
                    general_utils.clean(self.raw_dataset_paths[filename], filename)
                    
        print("Preprocess complete...")  
    
    def get_kaggle_dataset(self, filename):
        path = kagglehub.dataset_download(f"{filename}")

        print("Path to dataset files:", path) 

                        
                        

            
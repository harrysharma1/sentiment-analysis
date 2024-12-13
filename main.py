import os
import requests


class Preprocessing():
    def __init__(self):
        self.url = {
            "hate-speech.csv":"https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv",
        }
        
        
        if not os.path.exists(f"{os.getcwd()}/datasets"):
            os.mkdir(f"{os.getcwd()}/datasets")
    
    def get_datasets(self):
        for filename, url in self.url.items():
            try:
                response = requests.get(url)
                path = os.path.join(os.getcwd(), 'datasets', filename) 
                with open(path, 'wb') as file:
                    file.write(response.content)
                print(f"Data saved to: {path}")
            except requests.HTTPError as e:
                match e.response.status_code:
                    case 404:
                        print(f"Error:\nDataset not found.\nStatus code: {e.response.status_code}.")
                    case _ :
                        print(f"Error:\nUnkown.\nStatus code: {e.response.status_code}")
            except requests.ConnectionError as e:
                print(f"Error:\nHostname was not resolvable.\nFull error: {e}")

                        
                        
a = Preprocessing()
a.get_datasets()

            
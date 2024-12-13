import os

class Preprocessing():
    def __init__(self, url):
        self.url = url
        
        if not os.path.exists(f"{os.getcwd()}/datasets"):
            os.mkdir(f"{os.getcwd()}/datasets")
import nltk
from nltk.corpus import stopwords
import re
import string

class Utilities():
    def __init__(self):
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words('english'))
    
    
    def remove_punctuation(self, text):
        return "".join([i for i in text if i not in string.punctuation])
    
    def tokenization(self, text):
        return text.split(' ')
    
    def clean_tokens(self, text):
        return [i for i in text if i != '']  
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
import string

class Utilities():
    def __init__(self):
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.stop_words = set(stopwords.words('english'))
        self.word_net_lemmatizer = WordNetLemmatizer()
    
    
    def remove_punctuation(self, text):
        return "".join([i for i in text if i not in string.punctuation])
    
    def tokenization(self, text):
        return text.split(' ')
    
    def clean_tokens(self, text):
        return [i for i in text if i != ''] 
    
    def remove_stop_words(self, text):
        return [i for i in text if i not in self.stop_words]
    
    def lemmatize(self, text):
        return [self.word_net_lemmatizer.lemmatize(i) for i in text]    
from src.utilities import *


class HateSpeechUtilities(Utilities):
    def __init__(self):
        super().__init__()
    
    def clean(self, tweet):
        tweet = str(tweet)

        tweet = re.sub(r'&#8221;', '', tweet)
        tweet = re.sub(r'https?:\/\/t\.co\/[a-zA-Z0-9]+','',tweet)
        tweet = re.sub(r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$', '', tweet) 
        tweet = re.sub(r'@[\w]+', '', tweet) 
        tweet = re.sub(r'RT', '', tweet)
        
        tweet = re.sub(r'&#\d+;', '', tweet) 
        tweet = re.sub(r'&amp;', '', tweet) 
        tweet = tweet.lstrip(' ')

        return tweet
    
    
from src.utilities import *


class HateSpeechUtilities(Utilities):
    def __init__(self):
        pass
    
    def clean_hate_speech(self, tweet):
        tweet = str(tweet)

        tweet = re.sub(r'&#8221;', '', tweet)
        tweet = re.sub(r'http:\/\/t\.co\/[a-zA-Z0-9]+','',tweet)
        tweet = re.sub(r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$', '', tweet) 
        tweet = re.sub(r'@[\w]+', '', tweet) 
        tweet = re.sub(r'RT', '', tweet)
        
        tweet = re.sub(r'&#8220;', '', tweet)
        tweet = re.sub(r'&#128553;', '', tweet)
        tweet = re.sub(r'&#128514;', '', tweet)
        tweet = re.sub(r'&#128517;', '', tweet)
        tweet = re.sub(r'&#8217;', '', tweet)
        tweet = re.sub(r'&#128175;', '', tweet)
        tweet = re.sub(r'&#128049;', '', tweet)
        tweet = re.sub(r'&#128533;', '', tweet)
        tweet = re.sub(r'&#128554;', '', tweet)
        tweet = re.sub(r'&#128527;', '', tweet)
        tweet = re.sub(r'&#128056;', '', tweet)
        
        tweet = tweet.lstrip(' ')

        return tweet
    
    
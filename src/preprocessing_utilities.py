from src.utilities import *


class HateSpeechUtilities(Utilities):
    def __init__(self):
        super().__init__()
    
    def clean_tweet(self, tweet):
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
    
    def clean(self, raw_path, filename):
        hate_speech = HateSpeechUtilities()
        data_frame = pd.read_csv(raw_path)
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: hate_speech.clean_tweet(x))
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
        return (clean_csv_path, 'tweet')
         
class SouthParkUtilities(Utilities):
    def __init__(self):
        super().__init__()
        
    def clean(self, raw_path, filename):
        data_frame = pd.read_csv(raw_path)
        data_frame['Line'] = data_frame['Line'].apply(lambda x: self.remove_punctuation(x))
        data_frame['Line'] = data_frame['Line'].apply(lambda x: x.lower())
        data_frame['Line'] = data_frame['Line'].str.replace('\n','')
        data_frame['Line'] = data_frame['Line'].apply(lambda x: self.tokenization(x))
        data_frame['Line'] = data_frame['Line'].apply(lambda x: self.clean_tokens(x))
        data_frame['Line'] = data_frame['Line'].apply(lambda x: self.remove_stop_words(x))
        data_frame['Line'] = data_frame['Line'].apply(lambda x: self.lemmatize(x))
        
        clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
        data_frame.to_csv(clean_csv_path,index=0)
        print (f"Cleaned data saved to: {clean_csv_path}")
        return (clean_csv_path, 'Line')


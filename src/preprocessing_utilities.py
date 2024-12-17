from src.utilities import *


class GeneralPurposeUtilitiees(Utilities):
    def __init__(self):
        super().__init__()
    
    def is_likely_text_column(self, series):
        return series.apply(lambda x: isinstance(x, str)).mean() > 0.8
        
    def likely_text_column(self, series):
        text_columns = [col for col in series.columns if self.is_likely_text_column(series[col])]
        likely_text_column = [
            col for col in text_columns if series[col].apply(lambda x: len(x) if isinstance(x, str) else 0).mean() > 30
        ]
        if likely_text_column:
            for col in likely_text_column:
                print(f"Preview of text column: '{col}':")
        return col
    
    def clean(self, raw_path, filename):
        data_frame = pd.read_csv(raw_path)
        col = self.likely_text_column(data_frame)
        data_frame[col] = data_frame[col].apply(lambda x: self.remove_punctuation(x))
        data_frame[col] = data_frame[col].apply(lambda x: x.lower())
        data_frame[col] = data_frame[col].str.replace('\n',' ')
        data_frame[col] = data_frame[col].apply(lambda x: self.tokenization(x))
        data_frame[col] = data_frame[col].apply(lambda x: self.clean_tokens(x))
        data_frame[col] = data_frame[col].apply(lambda x: self.remove_stop_words(x))
        data_frame[col] = data_frame[col].apply(lambda x: self.lemmatize(x))
        
        clean_csv_path = os.path.join(os.getcwd(),'datasets/clean',f'{filename}.csv')
        data_frame.to_csv(clean_csv_path,index=0)
        print (f"Cleaned data saved to: {clean_csv_path}")
        print (f"As this is general purpose, please double check whether the title for text input column was correctly estimated.")
        return (clean_csv_path, col)
    
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
        data_frame = pd.read_csv(raw_path)
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: self.clean_tweet(x))
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: self.remove_punctuation(x))
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: x.lower())
        data_frame['tweet'] = data_frame['tweet'].str.replace('\n',' ')
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: self.tokenization(x))
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: self.clean_tokens(x))
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: self.remove_stop_words(x))
        data_frame['tweet'] = data_frame['tweet'].apply(lambda x: self.lemmatize(x))
        
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


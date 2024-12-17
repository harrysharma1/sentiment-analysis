from src.train import Preprocessing

a = Preprocessing()
# a.get_dataset(url="https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-1.csv", filename="skibidi_toilet")
# a.clean_dataset("skibidi_toilet")

a.get_kaggle_dataset("kaggle/hillary-clinton-emails")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pandas import *
import string

data = read_csv('data.csv', encoding='latin-1')

reviews = data.iloc[:, 1]

def stem(text):
    if text.endswith("ss") or text.endswith("ly") or text.endswith("ed"):
        text = text[:-2]
    elif text.endswith('ies'):
        text = text[:-3] + "y"
    elif text.endswith('s'):
        text = text[:-1]
    elif text.endswith('ing'):
        text = text[:-3]
    return text

positive_words = [
    "excellent", "amazing", "fantastic", "wonderful", "superb", "great", 
    "impressive", "delightful", "positive", "brilliant", "perfect", 
    "awesome", "outstanding", "enjoyable", "love", "recommend", 
    "satisfied", "best", "flawless", "beautiful", "worth", 
    "remarkable", "exciting", "refreshing", "exceptional", 
    "pleasant", "liked", "helpful", "terrific"
]

negative_words = [
    "terrible", "awful", "disappointing", "poor", "hate", "absurd",
    "horrible", "unsatisfactory", "worst", "annoying", "waste", 
    "flawed", "problem", "regret", "boring", "dreadful", 
    "frustrating", "unacceptable", "mediocre", "negative", 
    "dislike", "unimpressive", "confusing", "dull", "lacking",
    "unfortunate"
]

stemmed_positive_words = [stem(word) for word in positive_words]
stemmed_negative_words = [stem(word) for word in negative_words]

for r in reviews:
    tokens = word_tokenize(r)
    tokens = [word for word in tokens if word not in string.punctuation]
    stemmed = [stem(word.lower()) for word in tokens]
    print(stemmed)
    positive_count = sum(1 for word in stemmed if word in stemmed_positive_words)
    negative_count = sum(1 for word in stemmed if word in stemmed_negative_words)
    print(positive_count)
    print(negative_count)
    break
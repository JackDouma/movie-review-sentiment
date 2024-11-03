from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pandas import *
import string
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

data = read_csv('data.csv', encoding='latin-1')

reviews = data.iloc[:, 1]
labels = data.iloc[:, 0]

def stem(text):
    if text.endswith("ss") or (text.endswith("ly") and text != "only") or text.endswith("ed"):
        text = text[:-2]
    elif text.endswith('ies'):
        text = text[:-3] + "y"
    elif text.endswith('s'):
        text = text[:-1]
    elif text.endswith('ing'):
        text = text[:-3]
    elif text.endswith('ness'):
        text = text[:-4]
    return text

positive_words = [
    "excellent", "amazing", "fantastic", "wonderful", "superb", "great", 
    "impressive", "delightful", "positive", "brilliant", "perfect", 
    "awesome", "outstanding", "enjoyable", "love", "recommend", 
    "satisfied", "best", "flawless", "beautiful", "worth", 
    "remarkable", "exciting", "refreshing", "exceptional", 
    "pleasant", "liked", "helpful", "terrific", "good",
]

negative_words = [
    "terrible", "awful", "disappointing", "poor", "hate", "absurd",
    "horrible", "unsatisfactory", "worst", "annoying", "waste", 
    "flawed", "problem", "regret", "boring", "dreadful", 
    "frustrating", "unacceptable", "mediocre", "negative", 
    "dislike", "unimpressive", "confusing", "dull", "lacking",
    "unfortunate", "disappointed"
]

stemmed_positive_words = [stem(word) for word in positive_words]
stemmed_negative_words = [stem(word) for word in negative_words]

def getOnlyCount(tokens):
    return len(list((i for i, n in enumerate(tokens) if n == 'only')))

def getPositiveCount(tokens):
    return sum(1 for word in tokens if word in stemmed_positive_words)

def getNegativeCount(tokens):
    return sum(1 for word in tokens if word in stemmed_negative_words)

def getOnlySentiment(tokens):
    lines = []
    negatives = []
    result = 0
    only = list((i for i, n in enumerate(tokens) if n == 'only'))
    for o in only:
        try:
            lines.append([tokens[o + 1], tokens[o + 2], tokens[o + 3], tokens[o - 1]])
            negatives.append([tokens[o - 1]])
        except IndexError:
            continue
    for words in lines:
        for w in words:
            if w in stemmed_positive_words:
                if words[-1] == 'not':
                    result += 1
                else: 
                    result -= 1
            elif w in stemmed_negative_words:
                if words[-1] == 'not':
                    result -= 1
                else: 
                    result += 1
    return result

features = []

stop = False
for r in reviews:
    tokens = word_tokenize(r)
    tokens = [word for word in tokens if word not in string.punctuation]
    stemmed = [stem(word.lower()) for word in tokens]
    features.append([getPositiveCount(stemmed), getNegativeCount(stemmed), getOnlySentiment(stemmed)])


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print(accuracy_score(y_test, prediction))
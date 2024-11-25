import warnings
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import *
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import numpy as np

# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger_eng')

warnings.filterwarnings('ignore')

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))  # Create a set of English stopwords for filtering

vader = SentimentIntensityAnalyzer()

data = read_csv('prepro-data2.csv', encoding='latin-1')

labels = data.iloc[:, 0]
reviews = data.iloc[:, 1]
exclaims = data.iloc[:, 3]

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
    "pleasant", "liked", "helpful", "terrific", "good", "stunning"
]

negative_words = [
    "terrible", "awful", "disappointing", "poor", "hate", "absurd",
    "horrible", "unsatisfactory", "worst", "annoying", "waste", 
    "flawed", "problem", "regret", "boring", "dreadful", 
    "frustrating", "unacceptable", "mediocre", "negative", 
    "dislike", "unimpressive", "confusing", "dull", "lacking",
    "unfortunate", "disappointed", "rough"
]

adverbs = [
    "absolute", "amazing", "awful", "bare", "complete", "deep", "enormous",
    "entire", "especial", "extreme", "fabulous", "fair", "frightful", 
    "ful", "great", "hard", "high", "huge", "incredib", "insane", 
    "intense", "literal", "mild", "moderate", "particular", "phenomenal", 
    "pure", "quite", "rather", "real", "remarkab", "serious", "significant",
    "slight", "so", "somewhat", "strong", "surprising", "terrib", "thorough", 
    "total", "tremendous", "tru", "utter", "very", "virtual", "wild"
]

doc = nlp(" ".join(positive_words))
lemmatized_positive = [token.lemma_.lower() for token in doc]
doc = nlp(" ".join(negative_words))
lemmatized_negative = [token.lemma_.lower() for token in doc]

# is not actually used
def getOnlyCount(tokens):
    return len(list((i for i, n in enumerate(tokens) if n == 'only')))

def getPositiveCount(tokens):
    sum = 0
    for i in range(0, len(tokens)):
        if tokens[i] in lemmatized_positive:
            sum += 1
            if i != 0 and tokens[i - 1] in adverbs:
                sum += 1
    return sum

def getNegativeCount(tokens):
    sum = 0
    for i in range(0, len(tokens)):
        if tokens[i] in lemmatized_negative or "**" in tokens[i]:
            sum += 1
            if i != 0 and tokens[i - 1] in adverbs:
                sum += 1
    return sum

def getReverseSentiment(tokens):
    lines = []
    result = 0
    only = list((i for i, n in enumerate(tokens) if n == 'only'))
    for o in only:
        try:
            lines.append([tokens[o + 1], tokens[o + 2], tokens[o + 3], tokens[o - 1]])
        except IndexError:
            continue
    for words in lines:
        for w in words:
            if w in lemmatized_positive:
                if words[-1] == 'not':
                    result += 1
                else: 
                    result -= 1
            elif w in lemmatized_negative:
                if words[-1] == 'not':
                    result -= 1
                else: 
                    result += 1
    return result

# is not actually used
def getVaderScore(text):
    compound = vader.polarity_scores(text)['compound']
    if compound >= 0:
        return 0
    else:
        return 1

def getAdvToAdjRatio(text):
    tags = pos_tag(text)
    adjectives = 0
    adverbs = 0
    for word, tag in tags:
        if tag.startswith("JJ"):
            adjectives += 1
        elif tag.startswith("RB"):
            adverbs += 1
    return adverbs / adjectives if adjectives > 0 else 0

def getAverageVaderScore(words):
    total = 0
    i = 0
    for word in words:
        total += vader.polarity_scores(word)['compound']
        i += 1
    return total / i if i > 0 else 0

features = []

i = 0
for r in reviews:
    nouns = []
    adjectives = []
    verbs = []
    adverbs = []
    tokens = word_tokenize(r)
    tags = pos_tag(tokens)
    for word, tag in tags:
        if tag.startswith("NN"):
            nouns.append(word)
        elif tag.startswith("JJ"):
            adjectives.append(word)
        elif tag.startswith("V"):
            verbs.append(word)
        elif tag.startswith("RB"):
            adverbs.append(word)
    stemmed = [stem(word.lower()) for word in tokens]
    features.append([getPositiveCount(stemmed), getNegativeCount(stemmed), getReverseSentiment(stemmed), getAdvToAdjRatio(stemmed),
                     getAverageVaderScore(nouns), getAverageVaderScore(adjectives), getAverageVaderScore(verbs), getAverageVaderScore(adverbs),
                     len(stemmed), exclaims[i]])
    i += 1

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
# combined_features = hstack([features, tfidf_matrix])

# x_train, x_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.3, random_state=42)

classifiers = {
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": LinearSVC(dual=True)
}

scores = []
selectedScores = []

# fitting, predicting, and printing the accuracy for all the classifiers listed above
for name, classifier in classifiers.items():
    if isinstance(features, list):
        features = np.array(features)

    sfs1 = sfs(classifier, k_features=4, forward=False, verbose=0, scoring='accuracy')
    sfs1 = sfs1.fit(features, labels)

    feat_names = list(sfs1.k_feature_names_)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    combined_features = hstack([features, tfidf_matrix])

    x_train, x_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.3, random_state=42)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on " + name + ":", accuracy)
    scores.append(accuracy)

    selected_indices = [int(i) for i in feat_names]
    selected_features = features[:, selected_indices]
    combined_features = hstack([selected_features, tfidf_matrix])

    x_train, x_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.3, random_state=42)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on " + name + ":", accuracy)
    selectedScores.append(accuracy)

print(scores)
print(selectedScores)

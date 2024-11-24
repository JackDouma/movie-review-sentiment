# Program works as follows:
# 1. Preprocess reviews: Remove <br /> tags left in text of reviews, convert text to lowercase, remove punctuation from text, remove stop words from text, lemmatize words
#    Preprocessed data is saved out to a csv file as it takes a while to do.
# 2. Convert review text data into a TF-IDF feature matrix
# 3. Train various models
# 4. Make predictions with models and output accuracy scores
# 5. Visualize the data using a variety of methods



import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


nlp = spacy.load("en_core_web_sm")

# preprocessed_file = 'chris-preprocessed-data.csv'
preprocessed_file = 'prepro-data2.csv'


# 1. DATA PREPROCESSING

if os.path.exists(preprocessed_file):
    print("--> Loading preprocessed data...", end='')
    preprocessed_df = pd.read_csv(preprocessed_file)
    print("Complete.")

else:
    data = pd.read_csv('data.csv', encoding='latin-1')

    # console progress bar variables
    total_reviews_to_process = len(data.index)
    iteration = 1
    previous_print = -1


    preprocessed_reviews = []
    exclaims = []

    print("--> Preprocessing data:", end=' ')

    for review in data['review']:

        # progress bar
        percent_complete = (iteration / total_reviews_to_process) * 100
        if (int(percent_complete) != previous_print):
            print(f"{int(percent_complete)}%", end='...')
            previous_print = int(percent_complete)
            

        # remove br tags
        review_without_br = review.replace('<br />', '')
        review_without_exclaim = review.replace('!', ' ')

        doc = nlp(review_without_br)

        # tokenize reviews, lemmatize token, remove stop words, and remove punctuation
        filtered_tokens = [
            token.lemma_.lower()
            for token in doc
                if not token.is_stop and
                    not token.is_punct
        ]

        # append preprocessed review as a single string
        preprocessed_reviews.append(' '.join(filtered_tokens))

        # append amount of exclamation marks in the review
        exclaims.append(review_without_br.count('!'))

        # used in progress bar
        iteration += 1

    print("Complete.")

    # save preprocessed reviews to a CSV file
    print(f"--> Saving preprocessed reviews and values to {preprocessed_file}...", end='')
    preprocessed_df = pd.DataFrame({
        'value': data['value'],
        'review': preprocessed_reviews,
        'score': data['score'],
        'exclaim': exclaims
    })
    preprocessed_df.to_csv(preprocessed_file, index=False)
    print("Complete.")





# 2. FEATURE EXTRACTION USING TF-IDF

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_df['review'])


# 3. TRAIN MODELS

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, preprocessed_df['value'], test_size=0.2, random_state=42)  

# Train logistic regression model
print("--> Training logistic regression model...", end='')
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
print("Complete.")

# Train SVM model
print("--> Training SVM model...", end='')
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
print("Complete.")

# Train random forest model
print("--> Training random forest model...", end='')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Complete.")

# Train naive bayes model
print("--> Training naive bayes model...", end='')
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("Complete.")

# Train KNN model
print("--> Training KNN model...", end='')
knn_model = KNeighborsClassifier(n_neighbors=50)
knn_model.fit(X_train, y_train)
print("Complete.")





# 4. EVALUATE MODELS

models = {
    'Logistic Regression': logistic_model,
    'SVM': svm_model,
    'Random Forest': rf_model,
    'Naive Bayes': nb_model,
    'KNN': knn_model
}

print("\nMODEL EVALUATIONS:")
accuracies = {}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    
    print(f"- {model_name} model accuracy: {accuracy:.2f}")


# Model accuracy results:
# - Logistic Regression model accuracy: 0.89
# - SVM model accuracy: 0.88
# - Random Forest model accuracy: 0.86
# - Naive Bayes model accuracy: 0.87
# - KNN model accuracy: 0.80


# Model accuracy results without any of the preprocessing stuff (only SVM ends up being better somehow):
# - Logistic Regression model accuracy: 0.89
# - SVM model accuracy: 0.90
# - Random Forest model accuracy: 0.84
# - Naive Bayes model accuracy: 0.87
# - KNN model accuracy: 0.77


#####################
# 5. VISUALIZATIONS #
#####################

preprocessed_lengths = preprocessed_df['review'].str.len()
posNegPalette = {'Positive': 'green', 'Negative': 'red'}
positiveReviews = preprocessed_df[preprocessed_df['value'] == 1]
negativeReviews = preprocessed_df[preprocessed_df['value'] == 0]

##### DISTRIBUTION OF SCORES #####
sns.histplot(preprocessed_df['score'], bins=10, kde=True, color='purple')
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

##### EXCLAMATION MARK GRAPH ######

# create labels for graph
bins = [0,1,2,3,4,5,6,7,8,9,10, float('inf')]
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']

# separate reviews
negativeReviews.loc[:, 'exclaimBins'] = pd.cut(negativeReviews['exclaim'], bins=bins, labels=labels, right=False)
positiveReviews.loc[:, 'exclaimBins'] = pd.cut(positiveReviews['exclaim'], bins=bins, labels=labels, right=False)

negativeReviews.loc[:, 'reviewType'] = 'Negative'
positiveReviews.loc[:, 'reviewType'] = 'Positive'
combinedReviews = pd.concat([negativeReviews[['exclaimBins', 'reviewType']], positiveReviews[['exclaimBins', 'reviewType']]])

# count reviews in each bin for neg and pos reviews
reviewCounts = combinedReviews.groupby(['exclaimBins', 'reviewType']).size().reset_index(name='count')

# create graph
plt.figure(figsize=(10, 6))
sns.barplot(x='exclaimBins', y='count', hue='reviewType', data=reviewCounts, palette=posNegPalette)
plt.title('Distribution of Exclamation Marks in Positive and Negative Reviews')
plt.xlabel('Number of Exclamation Marks')
plt.ylabel('Frequency')
plt.legend(title='Review Type')
plt.show()

sns.scatterplot(x=preprocessed_df['score'], y=preprocessed_df['exclaim'], alpha=0.5)
plt.title('Score vs. Number of Exclamation Marks')
plt.xlabel('Score')
plt.ylabel('Number of Exclamation Marks')
plt.show()

##### CHARACTER LENGTH GRAPH #####

positiveReviews = preprocessed_df[preprocessed_df['value'] == 1]['review']
negativeReviews = preprocessed_df[preprocessed_df['value'] == 0]['review']

positiveLengths = positiveReviews.str.len()
negativeLengths = negativeReviews.str.len()

# create graph
plt.figure(figsize=(10, 6))
sns.kdeplot(positiveLengths, label='Positive Reviews', color='green', shade=True)
sns.kdeplot(negativeLengths, label='Negative Reviews', color='red', shade=True)
plt.title('Review Length Distribution in Positive and Negative Reviews')
plt.xlabel('Review Length')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, 3000)
plt.show()

avg_length_by_score = preprocessed_df.groupby('score')['review'].apply(lambda x: x.str.len().mean())
avg_length_by_score.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Average Review Length by Score')
plt.xlabel('Score')
plt.ylabel('Average Length')
plt.show()

##### TOP TERMS USED IN REVIEWS #####

## using tfidf
featureNames = tfidf_vectorizer.get_feature_names_out()
termCount = tfidf_matrix.mean(axis=0).A1

sortedIndices = termCount.argsort()[::-1][:25]

# get top terms and the scores
topTerms = [featureNames[i] for i in sortedIndices]
topScores = [termCount[i] for i in sortedIndices]

# create graph
plt.figure(figsize=(10, 6))
sns.barplot(x=topScores, y=topTerms, palette="viridis")
plt.title('Top 25 Terms')
plt.xlabel('Score')
plt.ylabel('Terms')
plt.show()

##### MODEL ACCURACY COMPARISON #####

plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.values()), y=list(accuracies.keys()), palette='coolwarm')
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.xlim(0.7, 1.0)
plt.show()

##### CONFUSION MATRIX OF EACH MODEL #####

# the following will provide true pos, true neg, false pos, false neg of each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    
##### MOST COMMON WORDS #####

def getMostCommonWords(reviews, top_n=20):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))
    word_matrix = vectorizer.fit_transform(reviews)
    
    # sum up all word amounts
    word_freq = word_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    word_count = dict(zip(words, word_freq))
    common_words = Counter(word_count).most_common(top_n)
    
    return common_words

positiveCommonWords = getMostCommonWords(positiveReviews)
negativeCommonWords = getMostCommonWords(negativeReviews)
positiveWords, positiveCount = zip(*positiveCommonWords)
negativeWords, negativeCount = zip(*negativeCommonWords)

# plot results

plt.figure(figsize=(16, 10))

# positive
plt.subplot(1, 2, 1)
sns.barplot(x=list(positiveCount), y=list(positiveWords), palette="Greens_d")
plt.title('Top 20 Most Common Words in Positive Reviews')
plt.xlabel('Word Count')
plt.ylabel('Words')

# negative
plt.subplot(1, 2, 2)
sns.barplot(x=list(negativeCount), y=list(negativeWords), palette="Reds_d")
plt.title('Top 20 Most Common Words in Negative Reviews')
plt.xlabel('Word Count')
plt.ylabel('Words')

plt.tight_layout()
plt.show()

bins = [1,2,3,4,7,8,9,10,float('inf')]
labels = ['1', '2', '3', '4', '7', '8', '9', '10']
preprocessed_df['score'] = pd.cut(preprocessed_df['score'], bins=bins, labels=labels)

for group in preprocessed_df['score'].unique():
    groupReviews = preprocessed_df[preprocessed_df['score_range'] == group]['review']
    commonWords = getMostCommonWords(groupReviews, top_n=20)
    print(f"Top words for {group} score range:", commonWords)


##### MOST COMMON PHRASES #####

def getMostCommonPhrases(reviews, top_n=20):
    # use CountVectorizer to count phrases
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    phraseMatrix = vectorizer.fit_transform(reviews)
    phrases = vectorizer.get_feature_names_out()
    
    phraseCount = phraseMatrix.sum(axis=0).A1
    topPhrases = sorted(zip(phraseCount, phrases), reverse=True)[:top_n]
    
    return topPhrases 


positiveTopPhrases = getMostCommonPhrases(positiveReviews)
negativeTopPhrases = getMostCommonPhrases(negativeReviews)

# Prepare data for plotting
positivePhrases, positiveCount = zip(*positiveTopPhrases)
negativePhrases, negativeCount = zip(*negativeTopPhrases)

# plot results

plt.figure(figsize=(16, 10))

# positive
plt.subplot(1, 2, 1)
sns.barplot(x=list(positivePhrases), y=list(positiveCount), palette="Greens_d")
plt.title('Top 20 Phrases in Positive Reviews')
plt.xlabel('Phrase')
plt.ylabel('Frequency')

# negative
plt.subplot(1, 2, 2)
sns.barplot(x=list(negativePhrases), y=list(negativeCount), palette="Reds_d")
plt.title('Top 20 Phrases in Negative Reviews')
plt.xlabel('Phrase')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
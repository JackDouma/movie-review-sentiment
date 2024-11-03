# Program works as follows:
# 1. Preprocess reviews: Remove <br /> tags left in text of reviews, convert text to lowercase, remove punctuation from text, remove stop words from text, lemmatize words
#    Preprocessed data is saved out to a csv file as it takes a while to do.
# 2. Convert review text data into a TF-IDF feature matrix
# 3. Train various models
# 4. Make predictions with models and output accuracy scores



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


nlp = spacy.load("en_core_web_sm")

preprocessed_file = 'chris-preprocessed-data.csv'


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

    print("--> Preprocessing data:", end=' ')

    for review in data['review']:

        # progress bar
        percent_complete = (iteration / total_reviews_to_process) * 100
        if (int(percent_complete) != previous_print):
            print(f"{int(percent_complete)}%", end='...')
            previous_print = int(percent_complete)
            

        # remove br tags
        review_without_br = review.replace('<br />', '')

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

        # used in progress bar
        iteration += 1

    print("Complete.")

    # save preprocessed reviews to a CSV file
    print(f"--> Saving preprocessed reviews and values to {preprocessed_file}...", end='')
    preprocessed_df = pd.DataFrame({
        'value': data['value'],
        'review': preprocessed_reviews
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
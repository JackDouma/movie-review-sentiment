# Train KNN classifiers with varying amount of neighbors, output accuracy
# Use bigrams from reviews as features


# RESULTS:

# KNN with 2 neighbours accuracy: 0.51
# KNN with 3 neighbours accuracy: 0.51
# KNN with 4 neighbours accuracy: 0.51
# KNN with 5 neighbours accuracy: 0.52
# KNN with 10 neighbours accuracy: 0.52
# KNN with 20 neighbours accuracy: 0.56
# KNN with 30 neighbours accuracy: 0.57
# KNN with 40 neighbours accuracy: 0.56
# KNN with 50 neighbours accuracy: 0.55


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv', encoding='latin-1')

vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(data['review']) 

y = data['value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for i in [2, 3, 4, 5, 10, 20, 30, 40, 50]:
    KNN = KNeighborsClassifier(n_neighbors=i)

    KNN.fit(X_train, y_train)

    y_pred = KNN.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f'KNN with {i} neighbours accuracy: {accuracy:.2f}')
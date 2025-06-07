import nltk
import random
import string
import pickle
from nltk.corpus import stopwords, movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

nltk.download('movie_reviews')
nltk.download('stopwords')

def clean_review(review):
    review = review.lower()
    stop_words = set(stopwords.words('english'))
    tokens = review.split()
    return " ".join([word for word in tokens if word not in stop_words and word not in string.punctuation])

documents = [(movie_reviews.words(fileid), category) 
             for category in movie_reviews.categories() 
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

reviews = [" ".join(words) for words, label in documents]
labels = [1 if label == "pos" else 0 for words, label in documents]

pipelines = {
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "SVM": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', SVC(probability=True, random_state=42))
    ])
}

X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

best_model, best_accuracy, best_model_name = None, 0, ""

for model_name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    if accuracy > best_accuracy:
        best_accuracy, best_model, best_model_name = accuracy, pipeline, model_name

print(f"\nBest Model: {best_model_name} ({best_accuracy:.2f} Accuracy)")

with open("sentiment_pipeline.pkl", "wb") as pipeline_file:
    pickle.dump(best_model, pipeline_file)

print(f"{best_model_name} Pipeline Saved!")
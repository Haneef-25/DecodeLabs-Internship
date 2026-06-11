import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("reviews.csv")

print(df.head())


# Text Preprocessing
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    words = [word for word in words
             if word not in stop_words]

    words = [lemmatizer.lemmatize(word)
             for word in words]

    return " ".join(words)

df["clean_review"] = df["review"].apply(preprocess)


# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000
)

X = tfidf.fit_transform(
    df["clean_review"]
)

y = df["sentiment"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Naive Bayes Model
model = MultinomialNB()

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print(classification_report(
    y_test,
    y_pred
))


# Custom Prediction
review = [
    "This movie was absolutely fantastic and amazing"
]

review_vector = tfidf.transform(review)

prediction = model.predict(review_vector)

print("Prediction:", prediction[0])
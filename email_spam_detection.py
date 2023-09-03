import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (Replace with your real dataset)
data = {
    'text': [
        'Congratulations! You won a prize!',
        'Please find the attached report.',
        'Earn money fast with this opportunity!',
        'Meeting agenda for next week.',
        'Free gift for you!',
        'Review and approve the document.',
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a vectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Build and train a spam classifier (Naive Bayes in this case)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

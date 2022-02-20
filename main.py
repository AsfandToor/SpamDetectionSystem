# Getting All Libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Accessing Dataframe
df = pd.read_csv('spam.csv')

# Converting the Categories to Spam
# 1 represents Spam and 0 represents Not-Spam
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Removing the Category Column
df.drop(columns=['Category'], inplace=True)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.2)

# Vectorizing the Emails
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_train = X_train.toarray()

# Training Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting Score
X_test_count = vectorizer.transform(X_test)
print("Model Score: {}".format(model.score(X_test_count, y_test)))

# Predicting Random Mail
# First one is Not Spam
# Second one is Spam
mails = [
    "Hey Andrewm, forward me your credentials",
    "Upto 20% discount for you, on exclusive brands. Don't miss the reward"
]
mails = vectorizer.transform(mails)
print(model.predict(mails))
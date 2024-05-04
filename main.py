import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # For Logistic Regression
from sklearn.svm import SVC  # For Support Vector Machines
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score # Proportion of predicted real news articles that are actually real
from sklearn.metrics import recall_score # Proportion of actual real news articles that are correctly predicted as real
from sklearn.metrics import f1_score # Harmonic mean of precision and recall, combining both metrics

# Download nltk stopwords (one-time download)
nltk.download('stopwords')

# load the true and fake news dataset
true = pd.read_csv('Datasets/True.csv')['text']
fake = pd.read_csv('Datasets/Fake.csv')['text']

true_df = pd.DataFrame({'text': true, 'label': 1})
fake_df = pd.DataFrame({'text': fake, 'label': 0})

# concatenate both dataset and shuffle the rows
df = pd.concat([true_df, fake_df], ignore_index = True).sample(frac = 1).reset_index(drop = True)

# display the dataset
print(df.head())

def clean_text(text):
  # Lowercase conversion
  text = text.lower()

  # Remove punctuation
  punctuations = "!\"#$%&()*+,/:;<=>?@[\\]^_`{|}~"
  for p in punctuations:
    text = text.replace(p, "")

  # Remove stop words
  stop_words = stopwords.words('english')
  text = ' '.join([word for word in text.split() if word not in stop_words])
  return text

def tokenize_text(text):
  """
  This function splits a text into a list of tokens (words).

  Args:
      text: String containing the text to be tokenized.

  Returns:
      A list of tokens (words).
  """
  tokens = word_tokenize(text)
  return tokens

vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

def tfidf_features(cleaned_texts):
  """
  This function converts a list of cleaned texts into TF-IDF features using scikit-learn's TfidfVectorizer.

  Returns:
      A sparse matrix of TF-IDF features.
  """
  features = vectorizer.fit_transform(cleaned_texts)
  return features

def train_test_split_data(X, y, test_size=0.2):
  """
  This function splits features (X) and labels (y) into training and testing sets.

  Args:
      X: A numpy array or pandas dataframe containing the features.
      y: A numpy array containing the labels.
      test_size: Float between 0.0 and 1.0 representing the proportion of data
                 allocated to the testing set (default: 0.2).

  Returns:
      Four numpy arrays: X_train, X_test, y_train, y_test representing the training
      and testing sets for features and labels respectively.
  """
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
  return X_train, X_test, y_train, y_test

def predict_on_test(model, X_test):
  """
  This function uses a trained model to predict labels for the testing data.

  Args:
      model: The trained model object.
      X_test: A numpy array or pandas dataframe containing the testing features.

  Returns:
      A numpy array containing the predicted labels for the testing data.
  """
  y_pred = model.predict(X_test)
  return y_pred

text = "This is a test sentence. It has punctuation and stop words!"
cleaned_text = clean_text(text)
print(cleaned_text)  # Output: test sentence. punctuation stop words
# Apply the function to the 'text' column
df_cleaned = df
df_cleaned['text'] = df_cleaned['text'].apply(clean_text)
print(df_cleaned)

cleaned_texts = df_cleaned['text'].tolist()  # Convert text column to a list

# Apply TF-IDF to the list of cleaned texts
X_tfidf = tfidf_features(cleaned_texts)
# (use X_tfidf instead of df_tfidf['text'])

text = "This is a sentence used for a test."
tokens = tokenize_text(text)
print(tokens)  # Output: ['This', 'is', 'a', 'sentence', 'used', 'for', 'a', 'test', '.']
df_tokens = df_cleaned
df_tokens['text'] = df_tokens['text'].apply(tokenize_text)

cleaned_texts = ["This is a news article about climate change.", "Another news report on politics."]
tfidf_features = tfidf_features(cleaned_texts)
print(tfidf_features)  # This will be a sparse matrix representation
df_tfidf = df_tokens
#df_tfidf['text'] = df_tfidf['text'].apply(tfidf_features)

X = X_tfidf
y = df_tfidf['label']
X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)

print(f"Training set size: {X_train.shape[0]}") # Training set size: 35918
print(f"Testing set size: {X_test.shape[0]}") # Testing set size: 8980

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Adjust max_iter as needed

# Train the model on the training data
model.fit(X_train, y_train)

y_pred = predict_on_test(model, X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}") # Accuracy: 0.9885

precision = precision_score(y_test, y_pred)  #, pos_label="real" "real" is the label for real news
print(f"Precision: {precision:.4f}") # Precision: 0.9865

recall = recall_score(y_test, y_pred) #, pos_label="real"
print(f"Recall: {recall:.4f}") # Recall: 0.9895

f1 = f1_score(y_test, y_pred) #, pos_label="real"
print(f"F1-score: {f1:.4f}") # F1-score: 0.9880

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
dataset=pd.read_csv('Hate_Speech_Detection.csv')
dataset


# In[3]:


dataset.isnull()


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


dataset["labels"] = dataset["class"].map({0: "Hate Speech",
                                          1: "Offensive Language",
                                          2: "No hate or offensive language"})
dataset


# In[7]:


data = dataset[["tweet","labels"]]
data


# In[11]:


import pandas as pd
import re
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove usernames (Twitter handles)
    text = re.sub(r'@[^\s]+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


# Apply preprocessing to the 'tweet' column
data.loc[:, 'preprocessed_tweet'] = data['tweet'].apply(preprocess_text)


# Display the preprocessed data
print(data[['tweet', 'preprocessed_tweet']].head())


# In[12]:


# Display the preprocessed data
print(data[['tweet', 'preprocessed_tweet']].head())


# In[15]:


x=np.array(data["tweet"])
y=np.array(data["labels"])
x           


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
X = cv.fit_transform(x)
X


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train


# In[19]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)


# In[20]:


from sklearn.metrics import confusion_matrix
cn = confusion_matrix(y_test,y_pred)
cn


# In[22]:


import seaborn as sns
import matplotlib.pyplot as ply
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(cn, annot=True, fmt=".0f", cmap="YlGnBu")


# In[24]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[30]:


sample = "Let's love each other "
sample = preprocess_text(sample)
print(sample)


# In[28]:


data1 = cv.transform([sample]).toarray()
data1


# In[31]:


dt.predict(data1)


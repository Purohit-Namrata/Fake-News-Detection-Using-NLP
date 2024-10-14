import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import re


fake_news=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Fake News Detection Using NLP/Fake_news/Fake.csv")
real_news=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Fake News Detection Using NLP/Fake_news/True.csv")

#print(fake_news.head())
#print(real_news.head())

fake_news['isTrue'] = 0
real_news['isTrue'] = 1

df=pd.concat([fake_news, real_news],axis=0)  

df=df.drop(['title','subject','date'],axis=1)
#print(df.head())

label_counts=df['isTrue'].value_counts()
#print(label_counts)

plt.bar(label_counts.index,label_counts.values,color=['red','blue'])
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Labels")
plt.xticks([0,1],['Fake','True'])
plt.show()

#Cleaning text to remove any unwanted strings
def preprocess_text(text):
    text=text.lower()
    text = re.sub(r'\[.*?\]|\W|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', ' ', text)
    return text

df['text']=df['text'].apply(preprocess_text)

X=df['text']
y=df['isTrue']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
vectorization=TfidfVectorizer()
X_train=vectorization.fit_transform(X_train)
X_test=vectorization.transform(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print(classification_report(y_test,prediction))
print("Prediction is ",accuracy_score(y_test,prediction)*100)


Output:
              precision    recall  f1-score   support

           0       0.99      0.98      0.98      4653
           1       0.98      0.99      0.98      4327

    accuracy                           0.98      8980
   macro avg       0.98      0.98      0.98      8980
weighted avg       0.98      0.98      0.98      8980

Prediction is  98.37416481069042


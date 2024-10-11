import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string

fake_news=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Fake News Detection Using NLP/Fake_news/Fake.csv")
real_news=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Fake News Detection Using NLP/Fake_news/True.csv")

print(fake_news.head())
print(real_news.head())

fake_news['isTrue'] = 0
real_news['isTrue'] = 1

df=pd.concat([fake_news, real_news],axis=0)

df=df.drop(['title','subject','date'],axis=1)
df.head()

label_counts = df['is_true'].value_counts()
plt.bar(label_counts.index, label_counts.values, color=['red', 'blue'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.xticks([0, 1], ['Fake', 'True'])
plt.show()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|\W|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(preprocess_text)


x = df["text"]
y = df["isTrue"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


lr = LogisticRegression()
lr.fit(xv_train,y_train)
pred_lr = lr.predict(xv_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_lr)



import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


df = pd.read_csv(r"C:\Users\my\Desktop\Datasets\stock sentiment\Data.csv",encoding="ISO-8859=1")


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index


for index in new_index:
    data[index] = data[index].str.lower()
    
headlines = []
for i in range(0,len(data.index)):
    headlines.append(" ".join(str(x) for x in data.iloc[i,0:25]))



countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)

randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


test_headlines = []
for i in range(0,len(test.index)):
    test_headlines.append(" ".join(str(x) for x in test.iloc[i,2:25]))
    
test_dataset = countvector.transform(test_headlines)
predict = randomclassifier.predict(test_dataset)


matrix = confusion_matrix(test['Label'],predict)
st.write(matrix)
score = accuracy_score(test['Label'],predict)
st.write(score)
report = classification_report(test['Label'],predict)
st.write(report)
 
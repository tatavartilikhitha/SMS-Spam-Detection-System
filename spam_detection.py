import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df=pd.read_csv("SMSSpamCollection",sep="\t",names=["label","message"])
df["label"]=df["label"].map({"ham":0,"spam":1})
X=df["message"]
y=df["label"]
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
user_msg=input("Enter a message: ")
msg_vector=vectorizer.transform([user_msg])
prediction=model.predict(msg_vector)
if prediction[0]==1:
    print("The message is classified as: SPAM")
else:
    print("The message is classified as: NOT SPAM")

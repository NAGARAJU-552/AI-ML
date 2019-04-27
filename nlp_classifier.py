     
import pandas as pd
df = pd.read_csv('./Consumer_Complaints.csv')

########## Data Preprocessing #############3
df.head()

df = df[pd.notnull(df['Consumer complaint narrative'])]

col = ['Product', 'Consumer complaint narrative']
df = df[col]
df.columns = ['Product', 'Consumer complaint narrative']


########### Model #############
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(df['Consumer complaint narrative'], df['Product'], random_state = 0)

######## Count Vectorizer Classifier ################
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
clf = MultinomialNB().fit(X_train_counts, y_train)

print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
print(clf.predict(count_vect.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))

X_test_counts = count_vect.transform(X_test)
y_pred = clf.predict(X_test_counts)


#print(metrics.classification_report(y_test, y_pred, target_names=df['Product'].unique()))
print(accuracy_score(y_test,y_pred))


######## TFIDF Classifier ################
tfidf_vect = TfidfVectorizer(binary=True, use_idf=True)
X_train_tfidf = tfidf_vect.fit_transform(X_train)

clf1 = MultinomialNB().fit(X_train_tfidf, y_train)
X_test_tfidf = tfidf_vect.transform(X_test)


y_pred1 = clf1.predict(X_test_tfidf)
#print(metrics.classification_report(y_test, y_pred, target_names=df['Product'].unique()))
print(accuracy_score(y_test,y_pred1))


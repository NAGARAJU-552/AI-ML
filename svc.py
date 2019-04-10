# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:54:58 2019

@author: sundar.p.jayaraman
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:58:28 2019

@author: sundar.p.jayaraman
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 06:55:58 2019

@author: sundar.p.jayaraman
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("./bank-additional_bank-additional-full.csv")

y_df = df[['y']]
X_df = df.drop(['y'],axis=1).copy()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for i in range(0,X_df.shape[1]):
    if X_df.dtypes[i]=='object':
        X_df[X_df.columns[i]] = le.fit_transform(X_df[X_df.columns[i]])

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
        
from sklearn.svm import SVC

svc = SVC(kernel='linear', class_weight='balanced')
svc.fit(X_train, y_train)

print("Accuracy score of Train:" , accuracy_score(y_train,svc.predict(X_train)))
print("Accuracy score of Test:", accuracy_score(y_test,svc.predict(X_test)))


svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

print("Accuracy score of Train:" , accuracy_score(y_train,svc.predict(X_train)))
print("Accuracy score of Test:", accuracy_score(y_test,svc.predict(X_test)))


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# In[4]:


url  = "iris.data"
names = ['sepal_length' , 'sepal_width' , 'petal_length' , 'petal_width' , 'class']

df  = pd.read_csv(url , names =names)


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


import matplotlib.pyplot as plt
df.hist()
plt.show()


# In[9]:


from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.show()


# In[11]:


X = np.array(df.drop(['class'] , 1))
print(X[0])
print(X[1])


# In[12]:


Y = np.array(df['class'])
print(Y)


# In[15]:


def getind(name):
    if(name == 'Iris-setosa'):
        return 1
    if(name == 'Iris-versicolor'):
        return 2
    if(name == 'Iris-virginica'):
        return 3
    return 4
Y = [getind(name) for name in Y]


# In[22]:


print(Y)


# In[32]:


X_train , X_test , Y_train , Y_test = train_test_split(X, Y , test_size = 0.2 ,random_state= 0)


# In[63]:


models = []
models.append(('KNN' , KNeighborsClassifier()))
models.append(('SVM' , SVC()))
models.append(('Logistic'  , LogisticRegression(solver = 'liblinear' , multi_class =  'ovr')))
for name , model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    print("%s : %f %f"  %(name , cv_results.mean() , cv_results.std()))
    model.fit(X_train , Y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(Y_test , predictions))
    print(classification_report(Y_test , predictions))


# In[ ]:





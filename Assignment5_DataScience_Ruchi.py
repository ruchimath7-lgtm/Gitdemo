#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Case Study on Titanic Dataset
# 

# In[1]:


import pandas as pd
data=pd.read_csv("C:\\Users\\Ruchi Goyal\\Downloads\\titanic.csv")
data.head()


# In[2]:


data.shape


# In[4]:


data1=data.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin"],axis=1)
data1.head()


# In[5]:


data1["Embarked"].unique()


# In[6]:


data1["Sex"]=data1["Sex"].map({"male":0,"female":1})
data1["Embarked"]=data1["Embarked"].map({"S":0,"C":1,"Q":2})
data1.head()


# In[7]:


data1.isnull().sum()


# In[9]:


X = data1.drop(["Survived","Embarked"], axis=1)
y = data1["Survived"]
X.head()


# In[10]:


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy='mean')  
X_imputed = imputer.fit_transform(X)
X_imputed[:10]


# In[11]:


import numpy as np
np.isnan(X_imputed).any()


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)


# In[13]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


# In[14]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(X_test)

print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))


# ## Summary
# 
# ### Out of 268 predictions, around 78% predictions are correct which is a good accuracy score.
# 
# ### Some room for improvement, especially in distinguishing class 1

# 
# 
# ## True Negatives (TN = 126): Out of 268, 126 correctly predicted class 0
# 
# ## False Positives (FP = 31): Out of 268, 31 Class 0 misclassified as class 1
# 
# ## False Negatives (FN = 29): Out of 268, 29 Class 1 misclassified as class 0
# 
# ## True Positives (TP = 82): Out of 268, 82 Correctly predicted class 1

# In[ ]:





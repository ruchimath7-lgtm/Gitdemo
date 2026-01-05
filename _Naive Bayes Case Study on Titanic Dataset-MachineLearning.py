#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
data=pd.read_csv("C:\\Users\\Ruchi Goyal\\Downloads\\titanic.csv")


# In[26]:


data.head()


# In[49]:


data.shape


# In[50]:


data["Embarked"].isnull().sum()


# In[41]:


data1=data.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin"],axis=1)
data1.head()


# In[42]:


data1["Embarked"].unique()


# In[43]:


data1["Sex"]=data1["Sex"].map({"male":0,"female":1})
data1["Embarked"]=data1["Embarked"].map({"S":0,"C":1,"Q":2})
data1.head()


# In[46]:


data1.isnull().sum()


# In[51]:


data1.describe()


# In[52]:


mean_age=data1["Age"].mean()
data1["Age"]=data1["Age"].fillna(mean_age)
data1.isnull().sum()


# In[54]:


data1.describe()


# In[55]:


mode_Embarked=data1["Embarked"].mode()
data1["Embarked"]=data1["Embarked"].fillna(mode_Embarked)
df.isnull().sum()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size)


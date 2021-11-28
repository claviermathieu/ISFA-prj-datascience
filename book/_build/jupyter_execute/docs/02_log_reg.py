#!/usr/bin/env python
# coding: utf-8

# # Régression logistique

# ## Packages

# In[6]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

# Personnal lib
import lib.data as data


# ## Données

# In[3]:


path = "https://www.data.mclavier.com/prj_datascience/brut_train.csv"
X, y = data.import_data(path); X, y


# Pour le moment, les données sont importées directement d'un csv.

# In[30]:


train = pd.read_csv('train_log.csv')
train.head()


# In[33]:


X = train.iloc[:, :-1].to_numpy()
y = train.iloc[:, -1].to_numpy()


# ## Modèle

# In[34]:


clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])


# In[35]:


clf.score(X, y)


# In[38]:


clf.get_params()


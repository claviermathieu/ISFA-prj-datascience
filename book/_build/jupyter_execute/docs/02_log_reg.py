#!/usr/bin/env python
# coding: utf-8

# # Régression logistique

# ## Packages

# In[2]:


from sklearn.linear_model import LogisticRegression

# Personnal lib
import lib.data as data


# ## Données

# In[3]:


path = "https://www.data.mclavier.com/prj_datascience/brut_train.csv"
X, y = data.import_data(path); X, y


# ## Modèle

# In[5]:


clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])


# In[ ]:


clf.score(X, y)


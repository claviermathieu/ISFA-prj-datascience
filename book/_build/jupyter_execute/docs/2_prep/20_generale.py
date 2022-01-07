#!/usr/bin/env python
# coding: utf-8

# # Préparation des données

# ## Packages

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ## Importation

# In[2]:


train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_train.csv")
train.head()


# In[ ]:





scaler = StandardScaler()


# ## Transformation

# ### Vehicle_Dammage

# In[34]:


dict_cat = {'No' : 0, 'Yes' : 1}
new_damage = train.Vehicle_Damage.apply(lambda x : dict_cat[x])
train['Vehicle_Damage'] = new_damage

train.head()


# ### Gender

# In[35]:


dict_cat = {'Male' : 0, 'Female' : 1}
new_gender = train.Gender.apply(lambda x : dict_cat[x])

train['Gender'] = new_gender
train.head()


# ### Age

# In[36]:


ohe = OneHotEncoder(sparse=False)
one_hot_VAge = ohe.fit_transform(np.asarray(train.Vehicle_Age).reshape(-1,1))

less_than_one = [one_hot_VAge[i][1] for i in range(len(one_hot_VAge))]
one_to_two = [one_hot_VAge[i][0] for i in range(len(one_hot_VAge))]
two_and_more = [one_hot_VAge[i][2] for i in range(len(one_hot_VAge))]

train['VAge1'] = less_than_one
train['VAge2'] = one_to_two
train['VAge3'] = two_and_more

train.drop(columns = 'Vehicle_Age', inplace = True)
train.head()


# ### Policy_Sales_Channel

# In[37]:


train.Policy_Sales_Channel.value_counts(sort=True).head(5)


# On constate au profiling, qu'il y a principelement 4 grandes catégorie.

# In[39]:


dict_cat = {152 : 0, 26 : 1, 124 : 2}

def default_dict(x):
    if x in dict_cat:
        return dict_cat[x]
    else:
        return 3

new_damage = train.Policy_Sales_Channel.apply(lambda x : default_dict(x))
train['Policy_Sales_Channel'] = new_damage

train.head()


# In[40]:


train.Policy_Sales_Channel.value_counts(sort=True).head(5)


# In[43]:


train


# In[46]:


ohe = OneHotEncoder(sparse=False)
one_hot_VPolicy = ohe.fit_transform(np.asarray(train.Policy_Sales_Channel).reshape(-1,1))


train['VPolicy0'] = [one_hot_VPolicy[i][0] for i in range(len(one_hot_VPolicy))]
train['VPolicy1'] = [one_hot_VPolicy[i][1] for i in range(len(one_hot_VPolicy))]
train['VPolicy2'] = [one_hot_VPolicy[i][2] for i in range(len(one_hot_VPolicy))]
train['VPolicy3'] = [one_hot_VPolicy[i][3] for i in range(len(one_hot_VPolicy))]

train.drop(columns = 'Policy_Sales_Channel', inplace = True)
train.head()


# ## Export

# On déplace la variable target en fin de dataframe.

# In[53]:


Response = train.Response
train.drop('Response', axis = 1, inplace = True)
train['Response'] = Response


# Exportation des données

# In[55]:


train.to_csv('train_log.csv', index = False)


# ## Package data
# 
# Les fonctions de préparation des données présentées ci-dessous ont été regroupées dans un fichier data.py afin de permettre l'obtention rapide, et au bon format, des données pour tester différents modèles.

# 

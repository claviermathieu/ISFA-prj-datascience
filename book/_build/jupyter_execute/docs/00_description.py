#!/usr/bin/env python
# coding: utf-8

# # Description des données

# ## Packages

# In[4]:


import numpy as np
import pandas as pd
import pandas_profiling as pp


# ## Importation

# In[5]:


test = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_test.csv")
train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_train.csv")
train.head()


# ## Pandas profiling
# 
# Avant de commencer une analyse manuelle des variables, on utilise la librairie *pandas_profiling* pour avoir une première analyse rapide de notre jeu de données.

# In[10]:


profile = pp.ProfileReport(train, title = "ISFA - Groupe 1 | Insurance cross-selling")
profile.to_file("data_desc.html")


# Le rapport de *pandas_profiling* est <a href = "https://www.data.mclavier.com/prj_datascience/data_desc.html">disponible ici</a>.

# ## Types
# 
# 
# On identifie les types de chaque variable du jeu de données.

# In[7]:


train.dtypes


# ## Valeurs manquantes

# In[17]:


train.isna().sum()


# In[ ]:





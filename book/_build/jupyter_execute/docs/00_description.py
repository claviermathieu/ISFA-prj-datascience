#!/usr/bin/env python
# coding: utf-8

# # Description des données

# ## Packages

# Voici la liste des packages utilisés pour étudier les données.

# In[13]:


import pandas as pd
import pandas_profiling as pp

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Importation

# Dans un premier temps, nous importons les données.

# In[2]:


test = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_test.csv")
train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_train.csv")
train.head()


# ## Pandas profiling
# 
# Avant de commencer une analyse manuelle des variables, on utilise la librairie *pandas_profiling* pour avoir une première analyse rapide de notre jeu de données.

# In[18]:


profile = pp.ProfileReport(train, title = "ISFA - Groupe 1 | Insurance cross-selling")
profile.to_file("data_desc.html")


# Le rapport de *pandas_profiling* est <a href = "https://www.data.mclavier.com/prj_datascience/data_desc.html">disponible ici</a>.

# ## Types
# 
# 
# On identifie les types de chaque variable du jeu de données.

# In[3]:


train.dtypes


# ## Valeurs manquantes

# Nous vérifions qu'il n'y ait pas de données absentes.

# In[4]:


train.isna().sum()


# ## Interprétations graphiques

# Par la suite, nous créons différents graphique pour essayer de mieux comprendre les données en comprenant l'impact marginal des variables.

# In[24]:


sns.catplot(data=train, kind="violin", x="Vehicle_Age", y="Annual_Premium", hue="Vehicle_Damage", split=True, height=6, aspect=16/9)


# On remarque que la distribution des primes est significativement différentes pour les véhicules ayant plus de deux ans d'age en fonction qu'ils aient déjà eu ou non un accident.

# In[25]:


sns.boxplot(data = train[train["Annual_Premium"] < 60000], x="Vehicle_Age", y="Annual_Premium", hue="Response")


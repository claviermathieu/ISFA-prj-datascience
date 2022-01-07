#!/usr/bin/env python
# coding: utf-8

# # Description des données

# Dans un premier temps, nous cherchons à comprendre le jeu de données mis à disposition.

# ## Packages

# Voici la liste des packages utilisés pour étudier les données.

# In[18]:


import pandas as pd
import pandas_profiling as pp


# In[19]:


# Autres packages outils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Importation

# Importation des données brutes.

# In[20]:


train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_train.csv")
train.head()


# ## Pandas profiling
# 
# Avant de commencer une analyse manuelle des variables, nous utilisons la librairie *pandas_profiling* pour avoir une première analyse rapide de notre jeu de données.

# In[21]:


profile = pp.ProfileReport(train, title = "ISFA - Groupe 1 | Insurance cross-selling")
profile.to_file("data_desc.html")


# Le rapport de *pandas_profiling* est <a href = "https://www.data.mclavier.com/prj_datascience/data_desc.html">disponible ici</a>.

# ## Analyse manuelle

# ### Information

# Voicil la liste des variables présentent dans le jeu de données

# In[ ]:


train.info()


# Nous pouvons constater qu'il y a 66 641 lignes et qu'il n'y a aucune valeur manquante.

# In[ ]:


train.describe().round(2)


# ## Types
# 
# 
# On identifie les types de chaque variable du jeu de données.

# In[ ]:


train.dtypes


# ## Valeurs manquantes

# Nous vérifions qu'il n'y ait pas de données absentes.

# In[ ]:


train.isna().sum()


# ## Interprétations graphiques

# Par la suite, nous créons différents graphique pour essayer de mieux comprendre les données en comprenant l'impact marginal des variables.

# In[ ]:


sns.catplot(data=train, kind="violin", x="Vehicle_Age", y="Annual_Premium", hue="Vehicle_Damage", split=True, height=6, aspect=16/9)


# On remarque que la distribution des primes est significativement différentes pour les véhicules ayant plus de deux ans d'age en fonction qu'ils aient déjà eu ou non un accident.

# In[ ]:


sns.boxplot(data = train[train["Annual_Premium"] < 60000], x="Vehicle_Age", y="Annual_Premium", hue="Response")


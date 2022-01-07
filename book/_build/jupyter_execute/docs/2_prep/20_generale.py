#!/usr/bin/env python
# coding: utf-8

# # Préparation des données

# In[22]:


# Bloc import des précédents notebook ---



import numpy as np
import pandas as pd
import pandas_profiling as pp
# Autres packages outils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_train.csv")
train.head()


# En fonction des modèles utilisées, il faut réaliser différentes préparation des données.
# 
# Par exemple, pour une régression logistique il faut transformer les variable en One-Hot variable alors que pour un modèle xgboost ce n'est pas obligatoire.
# 
# Ainsi, ce notebook a pour objectif d'exporter les différentes base de données avec le formalisme nécessaire à chaque type de modèle qui seront appliqués par la suite.
# 
# <br><br>
# 
# **Traitement global**
# 
# Cependant, dans un premier temps, certains traitement sont communes à tous les formalismes. Nous réalisons donc un nettoyage et le regroupement de certaines variables afin d'obtenir une base de données plus lisible.
# 
# ```{note} 
# Le nettoyage et le regroupement des données résulte de la description statistique précédente.
# ```
# 
# <br><br>
# 
# **Pour rappel**, voici la base de donnée d'origine

# In[23]:


train.head(3)


# ## Renommage

# Nous remplaçons les *No* et *Yes* de la variable Vehicle_Damage par des booleans.

# In[24]:


dict_cat = {'No' : 0, 'Yes' : 1}
train.Vehicle_Damage.replace(dict_cat, inplace = True)


# Ensuite, nous remplaçons les *Male* et *Female* par respectivement 0 et 1.

# In[25]:


dict_cat = {'Male' : 0, 'Female' : 1}
train.replace(dict_cat, inplace = True)


# Nous avons constaté que la variable *Policy_Sales_Channel* comportait beaucoup de catégorie alors que seulement 3 catégories dominent toutes les autres.
# 
# 

# In[26]:


train.Policy_Sales_Channel.value_counts(sort=True).head(5)


# Après agrégation, nous obtenons seulement 4 catégories de taille relativement homogènes.

# In[27]:


dict_cat = {152 : 0, 26 : 1, 124 : 2}

def default_dict(x):
    if x in dict_cat:
        return dict_cat[x]
    else:
        return 3

new_damage = train.Policy_Sales_Channel.apply(lambda x : default_dict(x))
train['Policy_Sales_Channel'] = new_damage


train.Policy_Sales_Channel.value_counts(sort=True).head(5)


# Enfin, nous modifions la variable *Vehicle_Age* pour la transformer en variable numérique.

# In[30]:


dict_age = {'1-2 Year' : 1, '< 1 Year' : 0, '> 2 Years' : 2}
train.replace(dict_age, inplace = True)


# Finalement, on obtient une base données comportant uniquement des variables de types int64.

# In[38]:


train.head(3)


# ```{note} 
# Ceci n'est pas obligatoire pour tous les modèles mais c'est une bonne pratique pour limiter le risque d'erreur.
# ```

# ## Filtre

# Nous avons remarquer que certains âges étaient aberrant. Comme dans la base de données test l'âge maximum est de 84 ans, nous filtrons notre base de données d'entraînement à un âge proche : 85 ans.

# In[45]:


train = train[train.Age <= 85]


# Le reste de la base de données étant très propre, aucun autre filtre ne parait nécessaire. Les lignes représentant des personnes sans permis pourraient paraître superflux mais quelques individus (peut-être des voitures de collections) ont tout de même une assurance. Ces personnes là existent aussi dans la base de données test. Nous ne faisons donc aucune manipulation dessus.

# ## Export 1

# La base de données obtenue avec les manipulations précédentes est suffisante pour 6 modèles que nous allons étudier par la suite.
# 
# - SVC
# - CART
# - Random Forest
# - Neural Network
# - Gradient boosting
# - XGBoost
# 
# 
# <br>
# 
# Nous exportons donc cette base de données que nous nommons *train_v1* et qui est héberger [sur serveur](https://www.data.mclavier.com/prj_datascience/) pour facilité l'accessibilité.

# In[54]:


train.to_csv("train_v1.csv", index = False)


# In[66]:


train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/train_v1.csv")


# ## Export 2

# Pour la régression logistique, il est fortement recommender de OneHot au maximum les variables pouvant l'être.
# 
# Pour cela nous utilisons sklearn.preprocessing.

# In[67]:


from sklearn.preprocessing import OneHotEncoder


# Nous commençons par OneHot la variable Age.

# In[68]:


ohe = OneHotEncoder(sparse=False)
one_hot_VAge = ohe.fit_transform(np.asarray(train.Vehicle_Age).reshape(-1,1))

less_than_one = [one_hot_VAge[i][1] for i in range(len(one_hot_VAge))]
one_to_two = [one_hot_VAge[i][0] for i in range(len(one_hot_VAge))]
two_and_more = [one_hot_VAge[i][2] for i in range(len(one_hot_VAge))]

train['VAge1'] = less_than_one
train['VAge2'] = one_to_two
train['VAge3'] = two_and_more

train.drop(columns = 'Vehicle_Age', inplace = True)
train.head(3)


# Puis la variable *Policy_Sales_Channel*

# In[69]:


ohe = OneHotEncoder(sparse=False)
one_hot_VPolicy = ohe.fit_transform(np.asarray(train.Policy_Sales_Channel).reshape(-1,1))


train['VPolicy0'] = [one_hot_VPolicy[i][0] for i in range(len(one_hot_VPolicy))]
train['VPolicy1'] = [one_hot_VPolicy[i][1] for i in range(len(one_hot_VPolicy))]
train['VPolicy2'] = [one_hot_VPolicy[i][2] for i in range(len(one_hot_VPolicy))]
train['VPolicy3'] = [one_hot_VPolicy[i][3] for i in range(len(one_hot_VPolicy))]

train.drop(columns = 'Policy_Sales_Channel', inplace = True)
train.head(3)


# La base de données pour la régression logistique peut à présent être exportée.
# 
# 
# <br>
# 
# Elle est nommée *train_v2* et est aussi hébergée [sur un serveur](https://www.data.mclavier.com/prj_datascience/) pour facilité l'accessibilité.

# In[70]:


train.to_csv("train_v2.csv", index = False)


# ## Conclusion

# Avec la création des bdd *train_v1.csv* et *train_v2.csv* le plus gros nettoyage a été réalisé.
# 
# Il n'y aura plus que quelques travaux de pre-processing de format en fonction des modèles implémentés.

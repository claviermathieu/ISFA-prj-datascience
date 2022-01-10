#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Bloc non affiché

import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score, matthews_corrcoef, make_scorer

from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import xgboost as xgb

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 14, 6


from sklearn.ensemble import RandomForestClassifier




def result_model(model,X,Y, mat = True, f1=True) :
    Y_model =model.predict(X)
    if f1:
        f1_scor = f1_score(Y,Y_model)
        print('Le f1 score vaut',f1_scor)
    
#     score = cross_val_score(model,X,Y,cv=5,scoring = make_scorer(f1_score))
#     print('F1 cross validé :', np.mean(score))
    
    if mat:
    # Matrice de confusion
        cm_model = confusion_matrix(Y, Y_model)
        plt.rcParams['figure.figsize'] = (5, 5)
        sns.heatmap(cm_model, annot = True)
        plt.title(str(model))
        plt.show()
    


# # Modèle finale

# ## Téléchargement des données

# In[3]:


train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/train_v1.csv")


# ## Pre-processing

# On sépare dans un premier temps les variables explicatives et la variable à expliquer.

# In[4]:


X = train.drop(columns='Response')
Y = train['Response']


# Le modèle final sera entrainé sur l'intégralité de la base que nous possédons. Mais actuellement, nous souhaitons mesure le caractère prédictif de nos données et donc pour éviter l'overfitting, nous séparons tout de même nos données.

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size = 0.85)


# ## Modèle

# Au final, nous avons tester 7 modèles. Voici leur F1-Score respectif :
# - **Logistique** (sans tuning) : 0.22
# - **SCV** (sans tuning) : 0.13
# - **CART** (sans tuning) : 0.43
# - **Random Forest** (avec tuning) : 0.55
# - **Réseau de neuronnes** (sans tuning) : 
# - **Gradient Boost** (sans tuning): 0.39
# - **XGBoost** (avec tuning) : 0.54

# Nous utilisons donc le **Random Forest** pour nos prévisions finales. Nous avons sans doute été chanceux avec la méthode par tâtonnement car le random forest paraît très efficace avec des temps d'entrainement bien moindres.

# In[6]:


params_rf = {
    'min_samples_split': 0.11959494750571721, 
    'min_samples_leaf' : 0.08048576405844253,
    'min_impurity_decrease' : 0.030792701550521537, 
    'n_estimators' : 88, 
    'class_weight' : 'balanced'
}


# Entrainement.

# In[7]:


rfc = RandomForestClassifier(**params_rf)
rfc.fit(X_train, Y_train)


# Résultat simple puis cross-validé.

# In[8]:


result_model(rfc, X_test, Y_test)


# In[9]:


scores_rf = cross_val_score(rfc, X, Y, cv=5, scoring='f1')
print("F1 moyen de %0.2f avec un écart type de %0.2f" % (scores_rf.mean(), scores_rf.std()))


# In[10]:


importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

feature_names = [i for i in X.columns]
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots(figsize = (10, 5))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# ## Export des prédictions

# Pour le rendu final, nous entrainons la base de données sur toute la bdd train.

# **Traitement de la bdd test**

# Nous appliquons le même traitement à la bdd test qu'à la bdd train.

# In[11]:


test = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_test.csv")
test.drop(columns='id', inplace = True)

dict_cat = {'No' : 0, 'Yes' : 1}
test.Vehicle_Damage.replace(dict_cat, inplace = True)

dict_cat = {'Male' : 0, 'Female' : 1}
test.replace(dict_cat, inplace = True)

dict_cat = {152 : 0, 26 : 1, 124 : 2}

def default_dict(x):
    if x in dict_cat:
        return dict_cat[x]
    else:
        return 3

new_damage = test.Policy_Sales_Channel.apply(lambda x : default_dict(x))
test['Policy_Sales_Channel'] = new_damage


dict_age = {'1-2 Year' : 1, '< 1 Year' : 0, '> 2 Years' : 2}
test.replace(dict_age, inplace = True)

X_to_predict = test


# In[12]:


X_to_predict.head(3)


# In[13]:


rfc_final = RandomForestClassifier(**params_rf)
rfc_final.fit(X, Y)


# Nous évaluons les prévisions finale :

# In[15]:


Y_predict = rfc_final.predict(X_to_predict)


# Puis, nous l'exportons sous le même format que la base de donnée X.

# In[16]:


Y_predict = pd.DataFrame(Y_predict)
Y_predict.rename(columns={0:'Response'}, inplace = True)
id_col = pd.read_csv("https://www.data.mclavier.com/prj_datascience/brut_test.csv", usecols=['id']).values
Y_predict['id'] = id_col
Y_predict = Y_predict[['id', 'Response']]

Y_predict.head()


# In[82]:


Y_predict.to_csv("groupe_1_predictions.csv", index = False)


# Vérification du fichier :

# In[19]:


pd.read_csv('https://www.data.mclavier.com/prj_datascience/groupe_1_predictions.csv')


# ## Conclusion

# Vous pouvez télécharger le fichier [groupe_1_prediction.csv](https://www.data.mclavier.com/prj_datascience/groupe_1_predictions.csv) pour évaluer le modèle.

# <br><br><br><br><br><br><br>

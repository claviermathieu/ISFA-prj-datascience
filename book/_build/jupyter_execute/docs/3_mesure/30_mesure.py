#!/usr/bin/env python
# coding: utf-8

# # Mesure

# In[6]:


# Non affiché
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## F1-Score

# D'après les consignes du projet. Le critère quantitatif appliqué pour évaluer le pouvoir prédictif de nos modèles sera le F1-score : 
# 
# $$
# F_1 = 2 \times \frac{\text{précision} \times \text{rappel}}{\text{précision} + \text{rappel}}
# $$
# 
# avec
# 
# $$
# \text{précision} = \frac{VP}{(VP + FP)} \qquad et \qquad \text{rappel} = \frac{VP}{(VP + FN)}
# $$

# <br>
# 
# Nous chercherons donc par la suite à optimiser le F1-Score.

# <br><br>
# 
# **Implémentation Python**
# 
# 
# Le package *sklearn* permet d'introduire beaucoup de mesures. La mesure f1_score est notamment disponible.

# In[1]:


from sklearn.metrics import f1_score


# ```{warning}
# Sous R, la fonction F1_Score() mesure par défaut le nombre de 0 correcte. Il faut rajouter l'argument positive = 1 pour obtenir l'équivalent du f1_score de sklearn.
# ```

# Voici d'autres mesures disponibles avec le package *sklearn* 
# 
# - matthews_corrcoef
# - roc_auc_score
# - confusion_matrix
# - accuracy_score
# - r2_score

# Nous utiliserons simplement en plus la matrice de confusion pour avoir un élément plus visuel.

# In[8]:


from sklearn.metrics import confusion_matrix


# ## Comparer les modèles

# Nous implémentons dans un premier temps une fonction permettant de facilement calculer le F1-Score en donnant à la fonction : 
# 
# - Un modèle
# - X test
# - Y test
# 
# Nous affichons par la même occation la matrice de confusion.

# In[7]:


def result_model(model,X,Y, mat = True, f1 = True) :
    Y_model = model.predict(X)

    if f1:
        f1_scor = f1_score(Y,Y_model)
        print('Le f1 score vaut',f1_scor)
        return(f1_scor)
    
    if mat:
    # Matrice de confusion
        cm_model = confusion_matrix(Y, Y_model)
        plt.rcParams['figure.figsize'] = (5, 5)
        sns.heatmap(cm_model, annot = True)
        plt.title(str(model))
        plt.show()
    


# On rappelle qu'une matrice de confusion donne abscisse les valeurs réelles et en ordonnées les prédites ce qui permet de connaître les faux positif, les vrais positifs etc...

# ## Conclusion
# 
# Maintenant que nous avons les outils pour mesurer l'efficacité de nos modèles et les bases de données préparées, nous pouvons passer à l'implémentation.
# 
# <br><br><br><br>

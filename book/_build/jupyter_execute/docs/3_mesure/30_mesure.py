#!/usr/bin/env python
# coding: utf-8

# # Mesure

# ## F1-Score

# D'après les consignes pour le projet. Le critère quantitatif appliqué pour évaluer le pouvoir prédictif de nos modèles sera le F1-score : 
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

# <br><br>
# 
# **Sur Python**
# 
# 
# Le package *sklearn* permet d'introduire beaucoup de mesures. La mesure f1_score est notamment disponible.

# In[1]:


from sklearn.metrics import f1_score, make_scorer


# Voici d'autres mesures disponibles avec le package *sklearn* 
# 
# - matthews_corrcoef
# - roc_auc_score
# - confusion_matrix
# - accuracy_score
# - r2_score

# ## Comparer les modèles

# In[3]:


a = 'bonjour'


# 

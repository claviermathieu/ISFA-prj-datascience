#!/usr/bin/env python
# coding: utf-8

# # Réseau de neurones

# In[1]:


from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

X = data.iloc[:, :-1]
Y = data['Response']

model = PCA(n_components=3)
model.fit(data.drop(['Response'], axis=1))
reduc = model.transform(data.drop(['Response'], axis=1))

X2_train, X2_test, Y2_train, Y2_test = model_selection.train_test_split(reduc,Y,train_size=60000)
#initialisation du classifieur
rna = MLPClassifier(hidden_layer_sizes=(3,),activation="logistic",solver="lbfgs")


# In[ ]:


#apprentissage
rna.fit(X2_train,Y2_train)

#affichage des coefficients
print(rna.coefs_)
print(rna.intercepts_)


# In[ ]:


#prédiction sur la base train après retraitements
pred = rna.predict(X2_test)
print(pred)


# In[ ]:


#mesure des performances
from sklearn import metrics
 
print(metrics.confusion_matrix(Y2_test,pred))
print("Taux de reconnaissance = " + str(metrics.accuracy_score(Y2_test,pred)))


# Taux de reconnaissance de 0.8

# *Perception simple*

# Nous importons les classes Sequential et Dense pour définir notre modèle et son architecture

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# La classe Sequential est une structure, initialement vide, qui permet de définir un empilement de couches de neurones

# In[ ]:


modelSimple=Sequential()


# Nous ajoutons une couche qui relie directement la couche d'entrée, input_dim: nombre de neurones = nombre de variables prédictives, avec la couche de sortie, units=1:une seule sorie puisque la variables cible est binaire, codée 1/0 avec une fonction d'activation sigmoïde.
# 

# In[ ]:


modelSimple.add(Dense(units=1,input_dim=3,activation="sigmoid"))


# Dense : tous les neurones de la couche précédente seront connectés à tous les neurones de la couche suivante.

# In[ ]:


print(modelSimple.get_config())


# En entrée du neurone de la couche de sortie, nous avons la combinaison linéaire suivante : 
# $$d(X)=a_0+a_1X_1+a_2X_2$$
# Après application de la fonction d'activation sigmöïde.
# 
# Nous avons en sortie du neurone de la couche de sortie :
# $$g(d)=\frac{1}{1+e^{-d}}$$
# $g(d)$ est une estimation de la probabilité conditionnelle $P(Y=pos/X_1,X_2)$ qui est déterminante dans les problèmatiques de classement.
# 
# L'étape suivante consiste à spécifier les caractéristiques de l'algorithme d'apprentissage : la fonction de perte à optimiser est l'entropie croisée binaire, elle correspond à la log-vraisemblance d'un échantillon où la probabilité conditionnelle d'appartenance aux classes est modélisée à l'aide de la loi binomiale. 
# 
# Adamax est l'algorithme d'optimisation, la métrique utilisée pour mesurer la qualité de la modélisation est le taux de reconnaissance ou taux de succès.

# In[ ]:


modelSimple.compile(loss="binary_crossentropy",optimizer ="Adamax",metrics=["accuracy"])


# Nous pouvons lancer l'estimation des poids synaptiques du réseau à partir des données étiquetées

# In[ ]:


#subdivision en apprentissage et test
from sklearn import model_selection
X2_train,X2_test,Y2_train,Y2_test = model_selection.train_test_split(reduc,Y,train_size=60000)
#apprentisage
modelSimple.fit(X2_train,Y2_train,epochs=150,batch_size=100)


# In[ ]:


print(modelSimple.get_weights())


# L'approche usuelle d'évaluation consiste à réaliser la prédiction sur l'échantillon test, puis à la contronter avec les valeurs observées de la variable cible. 

# In[ ]:


predSimple=modelSimple.predict(X2_test) 
classes_x=np.argmax(predSimple,axis=1)
#predSimple = modelSimple.predict_classes(X2_test)

print(metrics.confusion_matrix(Y2_test,classes_x))
score = modelSimple.evaluate(X2_test,Y2_test)
print(score)


# Graphique

# In[ ]:


import tensorflow as tf
def get_callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=70,restore_best_weights=True)]

history = modelSimple.fit(X2_train,Y2_train, validation_data=(X2_test, Y2_test), epochs=150, batch_size=100,   callbacks=get_callbacks(),verbose=0)

plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy')
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show() 


# *Perception multiple*

# Nous passons maintenant à un perceptron multicouche. Nous créons toujours structure Sequential, dans lequel nous ajoutons successivement deux objets Dense; le premier faisant la jonction entre la couche d'entrée et la couche caché.

# In[ ]:


modelMc = Sequential()
modelMc.add(Dense(units=6,input_dim=3,activation="sigmoid"))
modelMc.add(Dense(units=3,activation="sigmoid"))
modelMc.add(Dense(units=1,activation="sigmoid"))

modelMc.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
#apprentissage
modelMc.fit(X2_train,Y2_train,epochs=150,batch_size=10)

score = modelMc.evaluate(X2_test,Y2_test)
print(score)


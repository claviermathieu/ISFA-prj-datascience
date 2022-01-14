from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def F1(model,X,Y) :
    Y_model =model.predict(X)
    f1_scor = f1_score(Y,Y_model)
    return(f1_scor)


def result_model(model,X,Y, f1 = True, f1_aff = True, mat = True) :
    Y_model =model.predict(X)

    if f1:
        f1_scor = f1_score(Y,Y_model)
        if f1_aff:
            print('Le f1 score vaut',f1_scor)
        return(f1_scor)
        
        
    # Matrice de confusion
    if mat:
        cm_model = confusion_matrix(Y, Y_model)
        plt.rcParams['figure.figsize'] = (5, 5)
        sns.heatmap(cm_model, annot = True)
        plt.title(str(model))
        plt.show()
    

db <- read.csv("https://www.data.mclavier.com/prj_datascience/train_train_r.csv", sep=",")
rownames(db) = db$ind
db=db[c(2:ncol(db))]

head(db, 3)

dbr <- read.csv("https://www.data.mclavier.com/prj_datascience/train_test_r.csv", sep ="," )
rownames(dbr) = dbr$ind
dbr=dbr[c(2:ncol(dbr))]

head(dbr, 3)

db$Response = as.factor(db$Response)
dbr$Response = as.factor(dbr$Response)

library(randomForest)

rf <- randomForest(data=db, Response ~ ., ntree = 100, mtry = sqrt(ncol(db)-1),
                   nodesize = 1,maxnodes=NULL)

xtest = dbr[1:(ncol(db)-1)]
head(xtest, 3)

y_pred = predict(rf, xtest)

library(MLmetrics)

F1_Score(y_true = dbr$Response, y_pred = y_pred, positive = 1 )

train = pd.read_csv("https://www.data.mclavier.com/prj_datascience/train_v1.csv")

train.head(3)

X = train.drop(columns='Response')
Y = train['Response']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size = 0.85)

scaler=StandardScaler() 

X_scal_train = scaler.fit_transform(X_train)
X_scal_test = scaler.transform(X_test) 

X_scal_train = pd.DataFrame(X_scal_train,index= X_train.index)
X_scal_test = pd.DataFrame(X_scal_test,index= X_test.index)

def my_gsearch(params, param_grid, name_param):

    best_param = params[name_param]
    print(params)
    model = RandomForestClassifier(**params_rf).fit(X_train, Y_train)
    
    f1_max = result_model(model, X_test, Y_test, mat = False, f1_aff = False)

    for n_est in param_grid[name_param]:
        params[name_param] = float(n_est)
        model = RandomForestClassifier(**params)
        model.fit(X_train, Y_train)
        f1_temp = result_model(model, X_test, Y_test, mat = False, f1_aff = False)

        if f1_temp > f1_max:
            best_param = n_est
            f1_max = f1_temp

    params[name_param] = best_param
    print('Variable :', name_param, '(f1 :', f1_max, ', param :', best_param, ')')
    return(params, f1_max)

params_rf = {
    'min_samples_split': 0.11959494750571721, 
    'min_samples_leaf' : 0.08048576405844253,
    'min_impurity_decrease' : 0.030792701550521537, 
    'n_estimators' : 36, 
    'class_weight' : 'balanced', 
}

params_rf['n_estimators'] = 36

params_test = {
    'min_samples_split': np.arange(0.1, 1, step = 0.03), 
    'min_samples_leaf' : np.arange(0.01, 0.1, step = 0.0005),
    'min_impurity_decrease' : np.arange(0.01, 0.1, step = 0.005), 
    'n_estimators' : np.arange(1, 200, step = 5), 
    'class_weight' : 'balanced', 
}
params_rf, f1_max = my_gsearch(params_rf, params_test, 'min_samples_split')
params_rf, f1_max = my_gsearch(params_rf, params_test, 'min_samples_leaf')
params_rf, f1_max = my_gsearch(params_rf, params_test, 'min_impurity_decrease')
# params_rf, f1_max = my_gsearch(params_rf, params_test, 'n_estimators')

rfc = RandomForestClassifier(**params_rf)
rfc.fit(X_train, Y_train)
result_model(rfc, X_test, Y_test)

scores = cross_val_score(rfc, X_train, Y_train, cv=5, scoring='f1')
print("F1 moyen de %0.2f avec un écart type de %0.2f" % (scores.mean(), scores.std()))

rfc = RandomForestClassifier()
rfc.fit(X_scal_train, Y_train)
Y_rfc = result_model(rfc, X_scal_test, Y_test)

scores = cross_val_score(rfc, X_scal_train, Y_train, cv=5, scoring='f1')
print("F1 moyen de %0.2f avec un écart type de %0.2f" % (scores.mean(), scores.std()))

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

feature_names = [i for i in X.columns]
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots(figsize = (10, 5))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
Y_rfc = result_model(rfc, X_test, Y_test, mat = False)

maxi=F1(rfc, X_test, Y_test)
kmaxi=100

T=[]
test=[50,60,70,75,80,90,95,100,105,110,120]
for k in test:
    rfc = RandomForestClassifier(max_depth=k)
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    T.append(newF)
    if newF>maxi:
        maxi=newF
        kmaxi=k
print(kmaxi,maxi)
plt.plot(test,T)
plt.show()

n_estimators_int = kmaxi
print("Nous choisissons donc",kmaxi,"comme la valeur pour le paramètre n_estimators.")

maxi=F1(rfc, X_test, Y_test)
kmaxi=100

T=[]

test=[115,120,123,125,130]
for k in test:
    rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=k)
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    T.append(newF)
    if newF>maxi:
        maxi=newF
        kmaxi=k
print(kmaxi,maxi)
plt.plot(test,T)
plt.show()

max_depth_int=kmaxi
rfc = RandomForestClassifier(n_estimators=n_estimators_int, max_depth=max_depth_int)
rfc.fit(X_train, Y_train)
Y_rfc = result_model(rfc, X_test, Y_test)

maxi=F1(rfc, X_test, Y_test)
kmaxi = 2

T=[]
test=[2,3,4,5,10,20]
for k in test:
    rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=max_depth_int,min_samples_split=k)
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    T.append(newF)
    if newF>maxi:
        maxi=newF
        kmaxi=k
print(kmaxi,maxi)
plt.plot(test,T)
plt.show()

min_samples_split_int=kmaxi
print("Nous choisissons donc",kmaxi,"comme la valeur du paramètre min_samples_split.")

rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=max_depth_int,min_samples_split=min_samples_split_int)
rfc.fit(X_train, Y_train)
Y_rfc = result_model(rfc, X_test, Y_test)

maxi=F1(rfc, X_test, Y_test)
kmaxi = 1

T=[]
test=[1,2,3,4,5,10,20]
for k in test:
    rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=max_depth_int,min_samples_split=min_samples_split_int, min_samples_leaf=k)
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    T.append(newF)
    if newF>maxi:
        maxi=newF
        kmaxi=k
print(kmaxi,maxi)
plt.plot(test,T)
plt.show()

min_samples_leaf_int=kmaxi
print("Nous choisissons donc",kmaxi,"comme valeur du paramètre min_samples_leaf.")

rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=max_depth_int,min_samples_split=min_samples_split_int,min_samples_leaf=min_samples_leaf_int)
rfc.fit(X_train, Y_train)
Y_rfc = result_model(rfc, X_test, Y_test)

maxi=F1(rfc, X_test, Y_test)
kmaxi=0

T=[]
test=[0,0.0001,0.001,0.05,0.1]
for k in test:
    rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=max_depth_int,min_samples_split=min_samples_split_int, min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=k)
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    T.append(newF)
    if newF>maxi:
        maxi=newF
        kmaxi=k
print(kmaxi,maxi)
plt.plot(test,T)
plt.show()

min_impurity_decrease_int=kmaxi
print("Nous choisissons donc",kmaxi,"comme valeur du paramètre min_impurity_decrease.")

rfc = RandomForestClassifier(n_estimators=n_estimators_int,max_depth=max_depth_int,min_samples_split=min_samples_split_int,min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0)
rfc.fit(X_train, Y_train)
Y_rfc = result_model(rfc, X_test, Y_test)
maxi=F1(rfc, X_test, Y_test)
kmaxi=50

T=[]
test=[50,100,200,500]
for k in test:
    rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int, min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=k)
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    T.append(newF)
    if newF>maxi:
        maxi=newF
        kmaxi=k
print(kmaxi,maxi)
plt.plot(test,T)
plt.show()

n_estimators_int=kmaxi
print("Nous choisissons donc",kmaxi,"comme valeur du paramètre n_estimators.")

rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int, min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int)
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
print("avec auto F1=",newF)

rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int, min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int,max_features="sqrt")
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
print("avec sqrt F1=",newF)

rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int, min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int,max_features="log2")
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
print("avec log2 F1=",newF)

rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int,min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int,max_features="log2")
rfc.fit(X_train, Y_train)
Y_rfc = result_model(rfc, X_test, Y_test)
maxi=F1(rfc, X_test, Y_test)

rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int,min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int,max_features="log2")
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
print("avec auto F1=",newF)
rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int,min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int,max_features="log2",
                            class_weight="balanced")
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
print("avec balanced F1=",newF)
rfc = RandomForestClassifier(max_depth=max_depth_int,min_samples_split=min_samples_split_int,min_samples_leaf=min_samples_leaf_int,min_impurity_decrease=0,n_estimators=n_estimators_int,max_features="log2",
                            class_weight="balanced_subsample")
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
print("avec balanced_subsample F1=",newF)

def variation(param):
    modif=np.random.randint(0,5)
    if modif==3:
        param[3]=param[3]*np.random.uniform(0.5,1.5)+np.random.uniform()/100
    if modif==0 or modif==4:
        param[modif]=int(param[modif]*np.random.uniform(0.5,1.5))+1
    else:
        param[modif]=param[modif]*np.random.uniform(0.5,1.5)
    rfc = RandomForestClassifier(max_depth=param[0],min_samples_split=param[1],
                             min_samples_leaf=param[2],min_impurity_decrease=param[3],
                             n_estimators=param[4],max_features="log2", class_weight="balanced")
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    return(param,newF)


def variation2(param):
    modif=np.random.randint(0,5)
    if modif==3:
        param[3]=param[3]*np.random.uniform(0,10)+np.random.uniform()/100
    if modif==0 or modif==4:
        param[modif]=int(param[modif]*np.random.uniform(0.5,5))+1
    else:
        param[modif]=param[modif]*np.random.uniform(0.5,1.5)
    rfc = RandomForestClassifier(max_depth=param[0],min_samples_split=param[1],
                            min_samples_leaf=param[2],min_impurity_decrease=param[3],
                            n_estimators=param[4],max_features="log2", class_weight="balanced")
    rfc.fit(X_train, Y_train)
    newF= F1(rfc, X_test, Y_test)
    return(param,newF)

param = [max_depth_int,0.5,0.25,min_impurity_decrease_int,n_estimators_int]; param

T=[]
rip=0

rfc = RandomForestClassifier(max_depth=param[0],min_samples_split=param[1],
                             min_samples_leaf=param[2],min_impurity_decrease=param[3],
                             n_estimators=param[4],max_features="log2", class_weight="balanced")
rfc.fit(X_train, Y_train)
newF= F1(rfc, X_test, Y_test)
for k in range(10): #ou 100 voir 2000 pendant la nuit
    rip=rip+1
    print(k)
    nparam,nF=variation(param)
    T.append(nF)
    if nF>newF:
        newF=nF
        param=nparam
        print("+ Improvement")
        rip=0
    nparam,nF=variation2(param)
    T.append(nF)
    if nF>newF:
        newF=nF
        param=nparam
        print("+ Improvement")
        rip=0
    if rip==100:
        break


plt.plot(T)
plt.show()

param

rfc = RandomForestClassifier(max_depth=param[0],min_samples_split=param[1],
                             min_samples_leaf=param[2],min_impurity_decrease=param[3],
                             n_estimators=param[4],max_features="log2", class_weight="balanced")
rfc.fit(X_train, Y_train)
F1(rfc, X_test, Y_test)

scores = cross_val_score(rfc, X, Y, cv=5, scoring='f1')
print("F1 moyen de %0.2f avec un écart type de %0.2f" % (scores.mean(), scores.std()))

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 11:22:00 2018

@author: Aleksander


Классификация класса задач по времени решения задач, а не по значениям в ROI.


"""
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

os.chdir('E:\\diploma\\fmri2\\Kotovich_ROI\\time_classif')

list_dir = os.listdir()
results = []
for k in range(len(list_dir)):
    dat_list = []
    features = []
    sol_dat_m = np.array((0,0))
    num_masks=1
    with open(list_dir[k], 'r') as f:
        for line in f:
            dat_list.append([float(x) for x in line.split()])
        f.close()
        dat_m0 = np.array(dat_list)
        dat_m = np.zeros((60,num_masks))
        for i in range (60):
            for j in range(num_masks):
                if np.isnan(dat_m0[i,j]): dat_m0[i,j]=0
                dat_m[i,j] = dat_m0[i,j]
        sol_dat_m = dat_m
        for i in range(60):
            features.append(dat_m0[i,num_masks])

    sol_dat_m = sol_dat_m[:,:] 
    features = np.array(features)
    X_train, X_test, y_train, y_test = train_test_split(
       sol_dat_m, features, test_size=0.2, shuffle=True)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_traintr = scaler.fit_transform(X_train)
    X_testtr =  scaler.transform(X_test)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.neural_network import MLPClassifier

    parameters={#'cache_size': np.arange(50, 200, 50) }
    'n_estimators':np.arange(50,151,100),
    'max_depth':np.arange(10,100,60)}
    #'hidden_layer_sizes':np.arange(50,151,50)}
    #'max_features':np.arange(5,30,10)}
    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=4,verbose=3).fit(X_traintr, y_train)
    y_pred = clf.predict(X_testtr)
    results.append(f1_score(y_test,y_pred,average='micro'))
    #print(f1_score(y_test,y_pred,average='micro'))
    #sns.heatmap(confusion_matrix(y_test.astype(int),y_pred),annot=True)
    
plt.hist(results)
plt.show()
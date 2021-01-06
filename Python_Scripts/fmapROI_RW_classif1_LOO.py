# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:48:57 2019

@author: Aleksander


Скрипт для построения гистрограммы распределения процента правильного распознавания
правильности решения задач, а также средней ROC-AUC кривой для всех
испытуемых;
Строит также TSNE для визуального наблюдения различия в классах. 

*Выбор метода (Random Forest, MLP, LogReg...) осуществляется вручную - 
все перебираемые в каждом методе параметры закомментированы на строке 119. 
ЧТобы изменить метод, достаточно выбрать другой набор из параметров на строке 119;
Изменить название метода на строке 138; В некоторых случаях (для CV)
нужно будет раскомментировать строку 141. 

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from covariance_processing import PCA
from train_test_preparing import equalize

os.chdir(r'E:\diploma\fmri2\RWA_-1TR_classif')

list_dir = os.listdir()

#%%

X_emb = []
y_probas = []
results = []

k = 0
dat_list = []
with open(list_dir[k], 'r') as f:
    for line in f:
        dat_list.append([float(x) for x in line.split()])
    f.close()
dat_m0 = np.array(dat_list)    
        
num_masks = dat_m0.shape[1]  
scaler = StandardScaler()
        
for z in range(len(list_dir)//2): 
    if '0786' not in list_dir[z] and '0757' not in list_dir[z] \
                    and '0772' not in list_dir[z] and '0861' not in list_dir[z] \
                    and '0863' not in list_dir[z]:  
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        for idx, name in enumerate(list_dir):
            if '0786' not in name and '0757' not in name \
                    and '0772' not in name and '0861' not in name \
                    and '0863' not in name:  
                if name[2:] not in list_dir[z]:
                    with open(name, 'r') as f:
                        for line in f:
                            new = [float(x) for x in line.split()]
                            X_train.append(new)
                            if 'RA' in name:
                                y_train.append(1)
                            else: y_train.append(2)
                else: 
                    with open(name, 'r') as f:
                        for line in f:
                            new = [float(x) for x in line.split()]
                            X_test.append(new)
                            if 'RA' in name:
                                y_test.append(1)
                            else: y_test.append(2)
        
        
        for i, arr in enumerate(X_train):
            for j, el in enumerate(arr):
                if type(el) != float: X_train[i][j]=0
                
        for i, arr in enumerate(X_test):
            for j, el in enumerate(arr):
                if type(el) != float: X_test[i][j]=0
                
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        
        if 2 not in y_test or 1 not in y_test: continue
        
        X_train, y_train = equalize(X_train, y_train)
        X_test, y_test = equalize(X_test, y_test)
        
        transformer = KernelPCA(kernel = 'linear', n_components=10)
        scores = transformer.fit_transform(X_train)
        scores_test = transformer.transform(X_test)
        

        
        X_traintr = scaler.fit_transform(scores)
        X_testtr =  scaler.transform(scores_test)
        

        
        
        parameters={
        #     'n_estimators':np.arange(50,300,100),                  #RFC
        #     'max_depth':np.arange(10,100,60),
        #     'max_features':np.arange(5,10,10)}
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],            #SVC
        # 'C': [.001, .01, .1, 1],
        # 'probability': [True]}
        'Cs': [.001, .01, .1, 1.0],                                #LogisticRegression
        'max_iter': 300}
        #'hidden_layer_sizes': [[50,10], [100, 50], 50, 100, 150],  # MLPC
        #'alpha': [.001, .01, .1, 1.0],
        #'max_iter': [400]}
        # 'n_neighbors': [5,7],                                        # KNNC
        # 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        # 'p': [1,2]}
        
        
        clf = LogisticRegressionCV(**parameters, cv=5, 
                                  multi_class='ovr').fit(X_traintr, y_train)
        

        #clf = GridSearchCV(MLPClassifier(), parameters, cv=4,verbose=0).fit(X_traintr, y_train)
        y_pred = clf.predict(X_testtr)
        print(z, ' subject: {:.3f}'.format(f1_score(y_test,y_pred,average='micro')))
        print(f1_score(y_train, clf.predict(X_traintr)))
        y_test = np.array(y_test)
        fpr, tpr, thresholds = roc_curve(y_test-1, clf.predict_proba(X_testtr)[:,1])
        y_probas.append((fpr,tpr, roc_auc_score(y_test-1, clf.predict_proba(X_testtr)[:,1])))
        X_emb.append(TSNE(n_components=2).fit_transform(X_testtr))
        # plt.scatter(X_emb[-1][:,0].tolist(), X_emb[-1][:,1].tolist(), c=y_test, alpha=0.7)
        # plt.title(str(z))
        # plt.show()
        #sns.heatmap(confusion_matrix(y_test.astype(int),y_pred),annot=True)
        results.append(clf.score(X_testtr, y_test))

#%% 

        
X_full = np.concatenate((X_traintr, X_testtr), axis=0)
y_full = y_train + y_test.tolist()
for i, y in enumerate(y_full):
    if y==1: y_full[i] = 'b'
    elif y==2: y_full[i] = 'r'
X_tsne = TSNE(n_components=2).fit_transform(X_full) 
plt.scatter(X_tsne[:,0].tolist(), X_tsne[:,1].tolist(), c=y_full, alpha=0.5)
fp = mpatches.Patch(color='b', label='1st class')
sp = mpatches.Patch(color='r', label='2nd class')
plt.legend(handles=[fp,sp])
plt.title('Full set')
plt.show()

for fpr, tpr, _ in y_probas:
    plt.plot(fpr,tpr, alpha = 0.3, color='blue')
plt.title('Кривые ошибок')

### Теперь считаем усредненную кривую ошибок:
    
roc_line_fpr = []
roc_line_tpr = []

for arr in y_probas:
    roc_line_fpr.extend(arr[0])
    roc_line_tpr.extend(arr[1])

ftpr = zip(roc_line_fpr, roc_line_tpr)

dftr = {item: [] for item in roc_line_fpr}

for i, val in enumerate(roc_line_fpr):
    dftr[val].append(roc_line_tpr[i])

roc_line_fpr = []
roc_line_tpr = []

for key in dftr:
    roc_line_fpr.append(key)
    roc_line_tpr.append(np.mean(dftr[key]))

x = zip(roc_line_fpr, roc_line_tpr)
xs = sorted(x, key=lambda tup: tup[0])
roc_line_fpr = [x[0] for x in xs]
roc_line_tpr = [float('{:.5f}'.format(x[1])) for x in xs]

k = True
while k:
    k=False
    for i in range(len(roc_line_tpr)-1):
        if float('{:.4f}'.format(roc_line_tpr[i+1])) < float('{:.4f}'.format(roc_line_tpr[i])):
            w_val = (roc_line_tpr[i]*len(dftr[roc_line_fpr[i]]) +
                     roc_line_tpr[i+1]*len(dftr[roc_line_fpr[i+1]])) / (
                     len(dftr[roc_line_fpr[i]]) + len(dftr[roc_line_fpr[i+1]]))
            roc_line_tpr[i] = float('{:.5f}'.format(w_val))
            roc_line_tpr[i+1] = float('{:.5f}'.format(roc_line_tpr[i]))+0.00001
            k = True


plt.plot(roc_line_fpr, roc_line_tpr, label='Усреднённая кривая ошибок', color='red')
plt.plot([0,1], [0,1], '--', color='black', alpha=0.7)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()
plt.show()

res_hist = np.histogram(results, bins=15)

#%%


plt.hist(results, bins = 10)
plt.title('Гистограмма распределения ППР')
plt.show()
print('Средний результат: {:.3f}'.format(np.mean(results)))
print('Отклонение от среднего: {:.3f}'.format(np.std(results)))
res_roc = 0
for arr in y_probas:
    res_roc += arr[2]
res_roc /= len(y_probas)

print('Cредняя площадь под кривой ошибок: {:.3f}'.format(res_roc))

difs_fpr = np.diff(roc_line_fpr)
res_sum = 0

for i in range(len(difs_fpr)):
    res_sum += difs_fpr[i]*roc_line_tpr[i]

print('Площадь под усредненной кривой ошибок: {:.3f}'.format(res_sum))







    

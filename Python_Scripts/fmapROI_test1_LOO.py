# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:48:57 2019

@author: Aleksander


LOO по единственной компоненте (PCA). 

Результат:
    Гистограмма ППР
    ROC-AUC кривая + усредненная кривая ROC-AUC
    Confusion Matrix
    Дополнительно построена гистограмма значений этой компоненты, чтобы 
    визуализировать представление классов в этой компоненте.


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
from covariance_processing import PCA
from sklearn.decomposition import KernelPCA

os.chdir(r'E:\diploma\fmri2\Kotovich_ROI\5mm_std_maxHRF123')

list_dir = os.listdir()

X_emb = []
y_probas = []
results = []
y_test_all = []
y_pred_all = []
N_iters = 100

k = 0
dat_list = []
with open(list_dir[k], 'r') as f:
    for line in f:
        dat_list.append([float(x) for x in line.split()])
    f.close()
dat_m0 = np.array(dat_list)    
        
num_masks = dat_m0.shape[1]-1  
scaler = StandardScaler()
      
for z, _ in enumerate(list_dir): 
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
                if idx != z:
                    with open(name, 'r') as f:
                        for line in f:
                            new = [float(x) for x in line.split()]
                            if new[-1]<3:
                                X_train.append(new[:-1])
                                y_train.append(int(new[-1]))
                else: 
                    with open(name, 'r') as f:
                        for line in f:
                            new = [float(x) for x in line.split()]
                            if new[-1]<3:
                                X_test.append(new[:-1])
                                y_test.append(int(new[-1]))
    
        
        for i, arr in enumerate(X_train):
            for j, el in enumerate(arr):
                if type(el) != float: X_train[i][j]=0
                
        for i, arr in enumerate(X_test):
            for j, el in enumerate(arr):
                if type(el) != float: X_test[i][j]=0
                
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        
        transformer = KernelPCA(kernel = 'linear', n_components=1)
        scores = transformer.fit_transform(X_train)
        scores_test = transformer.transform(X_test)
        
        # scores = np.column_stack((scores, np.ones((scores.shape[0],1))))
        # scores_test = np.column_stack((scores_test, np.ones((scores_test.shape[0],1))))
        
        for q in range(N_iters):
            permut = np.random.choice(scores.shape[0], int(scores.shape[0]*0.8), replace = False)
            y_train_permut = y_train[permut]
            scores1 = scores[permut, :]
            X_traintr = scaler.fit_transform(scores1)
            X_testtr =  scaler.transform(scores_test)
            
            # X_traintr = scaler.fit_transform(X_train)
            # X_testtr = scaler.transform(X_test)
            
            
            parameters={
                # 'n_estimators':np.arange(50,300,100),                  #RFC
                # 'max_depth':np.arange(10,100,60),
                # 'max_features':np.arange(5,10,10)}
                # 'n_estimators':np.arange(50,300,100),                  #RFC_1_comp
                # 'max_depth':np.arange(1,50, 5),
                # 'max_features': [1]}
            'kernel': ['sigmoid'],            #SVC
            'C': [.01],
            'probability': [True]}
            # 'Cs': [.001, .01, .1, 1.0],                                #LogisticRegression
            # 'max_iter': 300}
            # 'hidden_layer_sizes': [[50,10], [100, 50], 50, 100, 150],  # MLPC
            # 'alpha': [.001, .01, .1, 1.0],
            # 'max_iter': [400]}
            # 'n_neighbors': [5,7],                                        # KNNC
            # 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            # 'p': [1,2]}
            
            
            # clf = LogisticRegressionCV(**parameters, cv=5, 
            #                           multi_class='ovr').fit(X_traintr, y_train_permut)
            
    
            clf = GridSearchCV(SVC(), parameters, cv=4,verbose=0).fit(X_traintr, y_train_permut)
            y_pred = clf.predict(X_testtr)
            print(list_dir[z][:-4], ', Step: ', q, ' Accuracy: {:.3f}'.format(f1_score(y_test,y_pred,average='micro')))
            y_test = np.array(y_test)
            fpr, tpr, thresholds = roc_curve(y_test-1, clf.predict_proba(X_testtr)[:,1])
            y_probas.append((fpr,tpr, roc_auc_score(y_test-1, clf.predict_proba(X_testtr)[:,1])))
            # X_emb.append(TSNE(n_components=2).fit_transform(X_testtr))
            # plt.scatter(X_emb[-1][:,0].tolist(), X_emb[-1][:,1].tolist(), c=y_test, alpha=0.7)
            # plt.title(str(z))
            # plt.show()
            # sns.heatmap(confusion_matrix(y_test.astype(int),y_pred),annot=True)
            results.append(clf.score(X_testtr, y_test))
            y_test_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

#%%        

X_full = np.concatenate((X_traintr, X_testtr), axis=0)
y_full = y_train_permut.tolist() + y_test.tolist()
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


CM = confusion_matrix(y_test_all,y_pred_all)/ np.sum(confusion_matrix(y_test_all,y_pred_all))
sns.heatmap(CM, annot=True, fmt='.2%', cbar=False)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion Matrix')
plt.show()


#%%

scores_all = np.concatenate((scores, scores_test))

y_all = y_train.tolist()+y_test.tolist()
scores_cl1 = []
scores_cl2 = []
for i in range(len(y_all)):
    if y_all[i] == 1:
        scores_cl1.append(scores_all[i][0])
    else: scores_cl2.append(scores_all[i][0])

plt.hist(scores_cl1, color='blue', label = 'class 1', alpha=0.5, bins=30)
plt.hist(scores_cl2, color='red', label = 'class 2', alpha=0.5, bins=30)
plt.title('Сравнение классов по единственной компоненте')
plt.xlabel('Значение компоненты')
plt.ylabel('Количество семплов')
plt.legend()
plt.show()
        
        

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:48:57 2019

@author: Aleksander

Скрипт для набора статистики по Проценту Правильного Распознавания по группе (НЕ LOO) для 
НЕ-функционального атласа (нет функции выбора области интереса, рассматриваются все).
N-iters - количество итераций проверки. На каждой итерации случайным образом разделяются 
тестовые и тренировочные выборки, и применяется ML алгоритм (выбираемый на строках 92-115).
Результат:
    
    Гистограмма ППР;
    Confusion matrix с TP, TN, FP, FN результатами в процентах.
    ROC-AUC кривые (и усредненная кривая).
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
import  train_test_preparing as ttp

os.chdir(r'C:\Users\aleks\OneDrive\Рабочий стол\МИФИ\ДИПЛОМ_МИФИ\code')
numROI1 = 0

os.chdir(r'E:/diploma/fmri2/Kotovich_ROI/5mm_std_maxHRF123')

list_dir = [item for item in os.listdir() if '.txt' in item]
features = []
sol_dat_m = np.array((0,0))

y_probas = []
N_iters = 100

####ЛИБО ЭТО (ЕСЛИ НЕ ФУНКЦИОНАЛЬНЫЙ АТЛАС)
k = 0
dat_list = []
with open(list_dir[k], 'r') as f:
    for line in f:
        dat_list.append([float(x) for x in line.split()])
    f.close()
dat_m0 = np.array(dat_list)
        
num_masks = dat_m0.shape[1]-1  

X = []
y = []
for idx, name in enumerate(list_dir):
    if '0786' not in name and '0757' not in name \
                    and '0772' not in name and '0861' not in name \
                    and '0863' not in name:  
            with open(name, 'r') as f:
                for line in f:
                    new = [float(x) for x in line.split()]
                    if new[-1]<3:
                        X.append(new[:-1])
                        y.append(int(new[-1]))

# X = np.column_stack((np.array(X),np.ones((len(X),1))))
X = np.array(X)
y = np.array(y)

results = []
y_test_all = []
y_pred_all = []

for z in range (N_iters): 
        
    X_train, X_test, y_train, y_test = ttp.train_test_split(
           X, y, test_size=0.2, shuffle=True, equal=True)
    
    X_train, y_train = ttp.equalize(X_train, y_train)
  
    
    scaler = StandardScaler()
    
    X_traintr = scaler.fit_transform(X_train)
    X_testtr =  scaler.transform(X_test)
    
    
    parameters={
    # 'n_estimators':np.arange(50,300,100), #RFC
    # 'max_depth':np.arange(10,100,60),
    # 'max_features':np.arange(5,10,10)}
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], #SVC
    'C': [.001, .01, .1, 1],
    'probability': [True]}
    # 'Cs': [.001, .01, .1, 1.0], #LogisticRegression
    # 'max_iter': 300}
    # 'hidden_layer_sizes': [[50,10], [100, 50], 50, 100, 150], # MLPC
    # 'alpha': [.001, .01, .1, 1.0],
    # 'max_iter': [400]}
    # 'n_neighbors': [5,7], # KNNC
    # 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    # 'p': [1,2]}


    # clf = LogisticRegressionCV(**parameters, cv=5,
    # multi_class='ovr').fit(X_traintr, y_train)
    
    
    clf = GridSearchCV(SVC(), parameters, cv=4,verbose=0).fit(X_traintr, y_train)
    y_pred = clf.predict(X_testtr)
    print('step ',z+1,' / ', N_iters, ': ',f1_score(y_test,y_pred,average='micro'))
    # print(clf.best_params_)
    y_test = np.array(y_test)
    fpr, tpr, thresholds = roc_curve(y_test-1, clf.predict_proba(X_testtr)[:,1])
    y_probas.append((fpr,tpr, roc_auc_score(y_test-1, clf.predict_proba(X_testtr)[:,1])))
    plt.plot(fpr,tpr, alpha = 0.05, color='blue')
    
    y_test_all.extend(y_test)
    y_pred_all.extend(y_pred)
    

    results.append(clf.score(X_testtr, y_test))



for fpr, tpr, _ in y_probas:
    plt.plot(fpr,tpr, alpha = 0.01, color='blue')
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

plt.hist(results, bins = 15)
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

#%%
CM = confusion_matrix(y_test_all,y_pred_all)/ np.sum(confusion_matrix(y_test_all,y_pred_all))
sns.heatmap(CM, annot=True, fmt='.2%', cbar=False)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion Matrix')
plt.show()

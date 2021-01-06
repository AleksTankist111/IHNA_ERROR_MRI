#-------------------------------------------------------------------------------
# Name:         Train_Test_Preparing
# Purpose:      Подготовка данных перед подачей на классификаторы и регрессоры
#               Разделение на тренировочные и тестовые данные.
#
# Author:      Alexander Skakun
#
# Created:     05.09.2020
# Copyright:   (c) Alexander Skakun 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

"""
Это вспомогательный файл. Часть функций из него используется в некоторых скриптах.
"""



import numpy as np
import numpy.random as rnd

class Train_Size_Exception(Exception):

    def __init__(self):
        self.txt = 'Недостаточно представителей одного или нескольких классов.'


def train_test_split(X, y, test_size = 0.2, equal = False, shuffle = False, random_seed = None):

    """
    Функция, разбивающая выборку на тестовую и тренировочную.

    Parameters:
        X:  numpy matrix OR list of lists OR list of numpy arrays. Матрица
            признаков, размера NxM, где N - количество семплов, M - количество
            признаков в каждом семпле.

        y:  list OR numpy array: Массив меток, размером NxA (обычно A=1).

        test_size: 0<float<=1 OR int>1; Размер тестовой выборки.
            Если задан float, то размер выборки определяется в процентном
            отношении (например, 0.2 это 20%), если задан int - то определяется
            количество элементов из N.
            DEAFULT: 0.2

        equal:  boolean; Параметр, показывающий на необходимость набора в
            тренировочную выборку одинакового количества представителей каждого
            класса. Если в данной выборке недостаточно представителей классов
            для создания равной выборки размером 1-test_size, вызывается
            исключение: Train_Size_Exception.
            DEAFULT: False

        shuffle:   boolean; Определяет, перемешивать данные перед разделением,
            или выбирать тренировочные данные из первых семплов, а тестовые -
            из оставшихся.
            DEAFULT: False

        random_seed:    int; Определяет случайность при перемешивании.
            DEAFULT: None

    Return:

        X_train, y_train, X_test, y_test: 4 numpy array, где X - матрицы признаков,
            y - соответсвующие им списки меток.

    """

    rnd.seed(random_seed)

    if shuffle:
        idxs = rnd.permutation(len(y))
        X = X[idxs]
        y = y[idxs]

    if type(test_size) == int:
        n_trs = (len(y)-test_size)
    else: 
        n_trs = int(len(y)*(1-test_size))
        if (len(y) - n_trs) % 2 == 1: n_trs +=1

    if not equal:
        X_train = X[:n_trs]
        y_train = y[:n_trs]
        X_test = X[n_trs:]
        y_test = y[n_trs:]
    else:
        count = count_classes(y)

        n_trs = (n_trs//len(count)) * len(count)

        for key, val in count.items():
            if val < n_trs//len(count)+1:
                raise Train_Size_Exception()

        labels = {}.fromkeys(y)

        for i, val in enumerate(y):
            if labels[val] != None: labels[val].append(i)
            else: labels[val] = [i]

        X_train = np.empty((0,X.shape[1]))
        y_train = []
        X_test = np.empty((0, X.shape[1]))
        y_test = []
        
        n_test = (len(y) - n_trs)//len(count)
        
        for k, v in labels.items():

            X_train = np.append(X_train, X[v[n_test:]], axis=0)
            X_test = np.append(X_test, X[v[:n_test]], axis=0)
            y_train.extend(y[v[n_test:]])
            y_test.extend(y[v[:n_test]])

        X_train = np.array(X_train)
        X_test = np.array(X_test)

    return X_train, X_test, y_train, y_test



def equalize(X, y, supplement = False):

    """
    Функция, уравнивающая выборки дискретных меток.

    Parameters:
        X:  numpy matrix OR list of lists OR list of numpy arrays. Матрица
            признаков, размера NxM, где N - количество семплов, M - количество
            признаков в каждом семпле.

        y:  list OR numpy array: Массив меток, размером Nx1.

        supplement: Способ уравнивания выборок по меткам:
            True: Ищется максимальное количество одинаковых меток, и все
                остальные метки дополняются своими копиями до этого размера
            False: Ищется минимальное количество одинаковых меток, и все
                остальные метки отбрасывают часть до этого размера.
            По умолчанию: False;

    Return:
        X_new, y_new: массивы нового размера (KxM и Kx1), где K - число семплов
            при условии, что меток каждого класса теперь одинаковое количество.
    """
    
    labels = {}.fromkeys(y)
    y = np.array(y)
    
    for i, val in enumerate(y):
        if labels[val] != None: labels[val].append(i)
        else: labels[val] = [i]

    if supplement:
        max_l = 0
        for key in labels:
            if len(labels[key]) > max_l:
                max_l = len(labels[key])

        for key, val in labels.items():

            idxs = rnd.choice(val, size = max_l-len(val), replace = True)
            X = np.row_stack((X, X[idxs]))
            y = np.row_stack((y,y[idxs]))

    else:
        min_l = len(y)
        for key in labels:
            if len(labels[key]) < min_l:
                min_l = len(labels[key])

        X_new = None
        y_new = None
        for key, val in labels.items():

            idxs = val[:min_l]
            if type(X_new) == type(None):
                X_new = X[idxs]
                y_new = y[idxs]
            else:
                X_new = np.row_stack((X_new, X[idxs]))
                y_new = np.concatenate((y_new,y[idxs]), axis=0)

        X = X_new
        y = y_new.tolist()

    return X, y


def count_classes(y):

    """
    Функция подсчета количества представителей каждого класса в выборке

    Parameters:
        y: массив Nx1 - массив меток

    Return:
        Dictionary: key - имя метки; value - количество этой метки в выборке

    """

    labels = {}.fromkeys(y)

    for i, val in enumerate(y):
        if labels[val] != None: labels[val] += 1
        else: labels[val] = 1

    return labels


if __name__ == '__main__':
    print(' Данный модуль реализует подготовку данных для работы с MLA, \n \
    например разделение на тест-трейн, уравнивание выборок, подсчет меток.')

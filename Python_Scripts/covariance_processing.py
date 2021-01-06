#-------------------------------------------------------------------------------
# Name:        covariance_processing
# Purpose:      Processing covariance for future main array reconstruction
#
# Author:      Alexander Skakun
#
# Created:     05.09.2020
# Copyright:   (c) Alexander Skakun 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------


"Это вспомогательный файл, некоторые функции из которого используются для расчета \
    вероятностей в основных скриптах"

import numpy as np
import matplotlib.pyplot as plt

### Инициализируем функцию подсчета ковариаций:

def transform(arrs):

    """
    transform(arrs)

    Функция, необходимая для приведения всех аргументов к распределению
    одинаковых параметров.

    Parameters:
        arrs: список списков или numpy матрица; Размер NxM, где N - количество
            независимых массивов (каналов), M - количество элементов в каждом
            массиве.

    Return:
        numpy матрица NxM, состоящая из массивов, приведенных к нормальному
            распределению с std=1, с сохранением mean у каждого массива.

    """

    arrs = np.array(arrs)

    for i, arr in enumerate(arrs):
        arrs[i] /= np.std(arr)

    return arrs


def covariance(x, y, dif):

    """
    Функция, считающая ковариацию между значением из массива y и
        соответствующим ему значением из массива x.

    Parameters:
        x - numpy array or list;  list of features; size - (1 x N),
            where N - number of features;

        y - numpy array or list;  list of results; size - (1 x N);

        dif - int - latency between feature influences on result and result


    Return: 1 number - covariance in delay = dif

    """

    x_reduced = np.array(x[:len(x)-dif])
    y_reduced = np.array(y[dif:])

    cov_value =(x_reduced-x_reduced.mean()) @ (y_reduced-y_reduced.mean()).T

    return cov_value


def cov_range(x, y,  max_dif, min_dif = 0):

    """
    Функция, считающая ковариации между значением из массива y и
        соответствующим ему значением из массива x; Соответствия
        перебираются в интервале от [min_dif, max_dif)

    Parameters:
        x - numpy array or list;  list of features; size - (1 x N),
            where N - number of features;

        y - numpy array or list;  list of results; size - (1 x N);

        min_dif - INT number. Minimum considered latency between feature,
            influences on result, and result.
            DEAFULT: 0

        max_dif - INT number. Maximum considered latency between feature,
            influences on result, and result.

    Return: list of numbers, with lenght = max_dif-min_dif

    """

    cov_graph = []

    for dif in range(min_dif, max_dif):
        cov_value = covariance(x, y, dif)
        cov_graph.append(cov_value)

    return cov_graph


def important_difs(cov_matrix, border, bias = False):

    """
    Parameters:
        cov_matrix: numpy матрица, список numpy массивов или список списков.
            Суммарный размер: (N x M), где N - количество массивов признаков,
            M - количество рассмотреных задержек

        border: float - граница значимого уровня ковариации; Всё, что выше -
            будет приниматься как значимое

        bias: array или list (Nx1), состоящий из int: смещение каждого из
            массивов признаков (если список ковариаций начинался не с нуля)
        По умолчанию: для каждого признака bias = 0


    Return: difs: list: index - номер массива признаков (по порядку cov_matrix),
                        value - список из int-чисел - значимых задержек.


    """

    if not bias:
        bias = [0]*len(cov_matrix)

    difs = []
    cov_matrix = np.abs(cov_matrix)
    for i, cov_graph in enumerate(cov_matrix):

        difs.append([])
        for j, cov in enumerate(cov_graph):
                if cov>border:
                    difs[i].append(j+bias[i])

    return difs


def get_matriсes(X, y, difs):

    """
    Функция, составляющая матрицу семплов для дальнейшей подачи на MLA.

    Parameters:
        X:  numpy array; Матрица NxM, где N - число каналов, M - число элементов
            в каждом канале.

        y: numpy array OR list: массив из N чисел.

        difs:   list; Список, размером N, состоящий из списков различной длины:
            каждый список содержит набор наиболее значимых в этом канале задержек.

    Return: samples, labels;
        samples: матрица из семплов - KxL, K - количество получившихся семплов;
            L - количество значимых признаков;
        labels: список размера K -  метки соответствующих семплов.

    """

    max_dif = max_delay(difs)
    samples = []
    labels = []
    for i in range(max_dif, len(y)):
        labels.append(y[i])
        samples.append([])
        for idx, channel in enumerate(difs):
            for val in channel:
                samples[-1].append(X[idx][i-val])

    return np.array(samples), labels

def PCA(X, n_components='all'):
    """
    Функция для проведения PCA.

    Parameters:
    X - array[N x M], where N - number of samples, M - number of dimentions

    additionally:

    n_components - int, number of necessary components
    OR float number between 0 and 1 - number of necessary components in a percent count
    OR str: 'all' - все компоненты будут выведены
    DEAFULT: 'all'

    Return: evalues, evectors: 2 numpy массива;

        evalues - массив NxL, L<=M, содержит собственные значения наиболее значимых
            компонент (определяется с помощью n_components) в каждом из семплов.
        evectors содержит соответсвующие собственные векторы.

    """

     # Строим матрицу вариаций (отклонений от среднего)
    X_var = X-X.mean(axis=0)
    # Строим матрицу ковариаций
    cov_matrix = 1/X.shape[0] * (X_var.T @ X_var)

    # Считаем собственные значения и собственные вектора
    evals, evectors = np.linalg.eigh(cov_matrix)
    
    # Теперь ищем variance explained. Firstly, count cumulative sum the eigenvalues
    # normalize by the sum of eigenvalues
    variance_explained =  np.cumsum(evals)/np.sum(evals)
    if type(n_components)==float:
        vmin = list(variance_explained >= n_components)
        max_idx = vmin.index(True) # Количество элементов, которые можно брать для восстановления
    elif n_components=='all': max_idx = 0
    else: max_idx = len(evals)-n_components
    
    #plt.plot(evals)
    #plt.show()
    
    
    scores = X_var @ evectors
    #     X_new = scores[:,max_idx:] @ evectors[:,max_idx:].T + X.mean(axis=0)

    return scores[:,max_idx:], evectors[:,max_idx:]

def reduce_delays(difs, border):

    """
    Функция, возвращающая список списков с задержками для каждого канала,
        в каждом из которых оставлены только задержки больше пороговой.

    Parameters:
        difs:   list; Список, размером N, состоящий из списков различной длины:
            каждый список содержит набор наиболее значимых в этом канале задержек.

        border: int: целое число - пороговая длина задержки.

    Return:
        difs_new: list - список из списков с задержками,
            каждая задержка >=border

    """

    difs_new = []
    for arr in difs:
        difs_new.append([])
        for v in arr:
            if v>=border: difs_new[-1].append(v)

    return difs_new


def max_delay(difs):

    """
    Функция, реализующая поиск и вывод максимальной найденной задержки:

    Parameters:
        difs: list: index - номер массива признаков,
                        value - список из int-чисел - значимых задержек.

    Return:
        Максимальное значение среди всех задержек в списке
            (Или -1, если список пуст)

    """
    max_dif = -1
    for v in difs:
        for dif in v:
            if dif>max_dif:
                max_dif = dif

    return max_dif


def min_delay(difs):

    """
    Функция, реализующая поиск и вывод минимальной найденной задержки:

    Parameters:
        difs: list: index - номер массива признаков,
                        value - список из int-чисел - значимых задержек.

    Return:
        Минимальное значение среди всех задержек в списке
            (Или np.inf, если список пуст)

    """
    min_dif = np.inf
    for v in difs:
        for dif in v:
            if dif<min_dif:
                min_dif = dif

    return min_dif

def difs_size(difs):

    """
    Функция, реализующая подсчет количества значимых задержек:

    Parameters:
        difs: list: index - номер массива признаков,
                        value - список из int-чисел - значимых задержек.

    Return:
        Количество признаков в списке

    """
    n = 0
    for v in difs:
        n += len(v)

    return n


if __name__ == '__main__':
    print ('    Данный модуль реализует подсчет ковариаций и поиск значимых \n \
    элементов, ковариации которых достаточно высоки. Также модуль имеет \n \
    функции подсчета максимальной задержки, количества задержек и \n \
    другие вспомогательные функции')



# IHNA_ERROR_MRI
Set of scripts for processing the fMRI data for the grant.

В большинстве скриптов \*.py написаны описания, как и где их менять для получения результата на вашей машине и т.д.

В каждом скрипте нужно будет менять пути к файлам с данными. Файлы с данными - это текстовые документы формата ma\*\*\*\*.txt, где \* - это цифра. Например, ma00786.txt

Путь к файлам это строка вида "C:\..."

В части скриптов можно менять применяемые для расчетов методы машинного обучения (LogReg, Random Forest, MLP,...). 
Для этого в скриптах заготовлены закомментированные строки с уже подготовленными параметрами.

2 файла из представленных - вспомогательны, и необходимы для работы большинства других скриптов. Это файлы 
*train_test_preparing.py
*covariance_processing.py

Эти файлы взяты из другого проекта, и необходимы только для подготовки данных перед подачей на MLA. 
В этих файлах реализован удобный PCA, а также чуть более гибкий чем в sklearn алгоритм рандомизации данных.

**Поместите эти файлы в папку с остальными скриптами для корректной работы последних.



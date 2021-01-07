Здесь описывается настройка матлаба для работы со скриптами, а также описания самих скриптов.\

1) Цитата Влада Балаева:

Вот два скрипта, которые у себя нашел по извлечению признаков и меток (extractAnagVector, AnalysAnagLog)
Использовать так: vec=extractAnagVector(subjectPath,fmriPrefix,AnatPrefix,fmriFilePrefix,maskDir)
По идее должен выдавать матрицу (N+1)xM где N - число масок, M - число предъявлений задач.
N+1 потому что еще метка (число от 1 до 4 по числу задач)
subjectPath - папка с сессиями испытемых, fmriPrefix - префикс фМРТ директории, AnatPrefix - префикс анатомической директории, fmriFilePrefix - префикс файлов фМРТ (в нашем случае это будет ResI - так по-умолчанию назваются файлы после регрессии, maskDir - директория с масками)
Тут может быть не очевидным зачем нужно директорию с масками использовать, если мы уже все спроецировали и это отдельные файлы в папке каждого испытуемого. Так вот в папке могут быть и другие маски (например маска желудочков, используемая при регрессии), а нужно брать только те, которые есть в maskDir.


2) regress_mot_ventr (и вспомогательный файл regress_wo_ventr_job) регрессирует от движений изображения

* Чтобы запустить регрессию, нужно в командной строке матлаба написать regress_mot_ventr(путь к папке с сессиями людей в одинарных кавычках,префикс функциональной папки в одинарных кавычках,TR,как хотите, чтобы называлась папка с регрессированными данными фМРТ в одинарных кавычках) итого 4 входных параметра. Например regress_mot_ventr('E:\control_group','REST_DF30_+_',2,'regressOfRestData')


3) projectMasks проецирует маски то, что называется в индивидуальное пространство

  * Чтобы запустить второй скрипт пишем  projectMasks(путь к папке с сессиями людей в одинарных кавычках,префикс функциональной папки в одинарных кавычках,префикс папки со структурными данными в одинарных кавычках и фигурных скобках,абсолютный путь к папке с масками). Например projectMasks('E:\control_group','REST_DF30_',{'T1_SAG_3D_ANATOMY_1X1X1_FAST_0002'},'C:\Users\IHNA\Documents\MATLAB\masks\')

Чтобы запустить нужно:

* Чтоб был установлен матлаб + некоторые стандартные его пакеты. Точно не знаю какие. Image Processing Toolbox и Signal Processing Toolbox полагаю.
* w Чтобы был скачен spm8 и прописан в Path. В матлабе не нужно подгружать библиотеки в каждом скрипте отдельно как в python например (import numpy например). Они подгружаются глобально. В меню есть опция set path, где нужно найти spm8 и подгрузить "with subfolders"
* Файлы, который здесь расположены, тоже должны лежать в какой-либо папке прописанной в Path. Например в папке матлаб в моих документах


Если возник вопрос, почему для структурной папки в фигурных скобках, то это сделано вот зачем:
Функциональные папки как правило имеют специфичные названия, а вот структурные нет. Отличаются только номером
Соответственно вместо префикса нужно писать название папки целиком вместе с номером, чтобы оно отличалось от других папок. Вот только если оператор томографа накосячил - номер может отличаться. У части испытуемых это может быть T1_SAG_3D_ANATOMY_1X1X1_FAST_0002 а у других T1_SAG_3D_ANATOMY_1X1X1_FAST_0003 и они не обработаются. А Фигурные скобки я сделал, чтобы можно было писать несколько вариантов названий папок с анатомией. Например так {'T1_SAG_3D_ANATOMY_1X1X1_FAST_0002', 'T1_SAG_3D_ANATOMY_1X1X1_FAST_0003'} - тогда обработаются папки и с таким названием папки и с другим. Еще бывает, что из двух анатомий снимают только одну. 
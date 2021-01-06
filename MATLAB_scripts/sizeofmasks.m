function [vec] = sizeofmasks(Pathn,fmriPrefix,AnatPrefix,fmriFilePrefix,maskDir)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%переходим в папку с масками, считываем все названи€ файлов типа .nii и
%.img
cd(maskDir);
maskFiles=cat(1,dir('*.nii'),dir('*.img'));
coords=struct;
vec=[];

%считываем абсолютные пути всех необходимых масок из анатомической директории
for i=1:length(maskFiles)
    maskFilesAbsPath{i}=strcat(maskDir,'\',maskFiles(i).name);
end
%читаем все маски и берем координаты единичек дл€ каждой
for j=1:length(maskFiles)
    Imask=spm_read_vols(spm_vol(maskFilesAbsPath{j}));
    [coords(j).NZE1,coords(j).NZE2,coords(j).NZE3]=ind2sub(size(Imask),find(Imask));
end

%переходим в папку с сесси€ми. подразумеваетс€, что в ней лежит
%лог-файл,которые ищетс€ и его им€ записываетс€ в logFile
cd(Pathn);
DD=dir('00*');
for zz=1:length(DD)
subjectPath = strcat(Pathn,DD(zz).name);
cd(subjectPath);
logFile=dir('*.log');
if length(logFile)~=1
    disp('More than one log file');
end
%из лога извлекаетс€ нужна€ информаци€ в структуру logData - там будут
%лежать времена начал стимула и их длительность через параметр RT (и то и
%другое не в секундах, а в сканах)
logData=analysAnagLog(logFile.name);
%ищем папки с префиксами fmriPrefix и AnatPrefix
fmriDir=dir(strcat(fmriPrefix,'*'));
anatDir=dir(strcat(AnatPrefix,'*'));
if length(fmriDir)~=1 || length(anatDir)~=1
    disp('Something wrong with fmri and/or anatomy directories');
end

%переходим в фћ–“ директорию и читаем все файлы расширени€ .nii и .img с
%префиксом fmriFilePrefix.
cd(fmriDir.name);
fmriFiles=cat(1,dir(strcat(fmriFilePrefix,'*.nii')),dir(strcat(fmriFilePrefix,'*.img')));
if isempty(fmriFiles) || length(fmriFiles)<logData.time(end)
    disp('not enough fmri files');
end

curMaskSize=zeros(1,length(maskFiles));
for j=1:length(maskFiles)
    V=spm_vol(fmriFiles(logData.time(3)+1).name);
    Ic=spm_sample_vol(V,coords(j).NZE1,coords(j).NZE2,coords(j).NZE3,1);
    curMaskSize(j)= sum(~isnan(Ic));
end
cMS='VoxelsInMask'+string(DD(zz).name)+'.txt';
cd(Pathn);

fff=fopen(cMS,'w');
for j=1:length(maskFiles)
    fprintf(fff, '%20s ----> ', maskFiles(j).name);
    fprintf(fff, '%d\r\n' , curMaskSize(j));
end
fclose(fff);
end
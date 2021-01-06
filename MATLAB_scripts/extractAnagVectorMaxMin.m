function [vec] = extractAnagVectorMaxMin(Pathn,fmriPrefix,AnatPrefix,fmriFilePrefix,maskDir)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%��������� � ����� � �������, ��������� ��� �������� ������ ���� .nii �
%.img
cd(maskDir);
maskFiles=cat(1,dir('*.nii'),dir('*.img'));
coords=struct;
vec=[];
%��������� � ����� � ��������. ���������������, ��� � ��� �����
%���-����,������� ������ � ��� ��� ������������ � logFile
cd(Pathn);
DD=dir('00*');
for zz=1:length(DD)
subjectPath = strcat(Pathn,DD(zz).name);
cd(subjectPath);
logFile=dir('*.log');
if length(logFile)~=1
    disp('More than one log file');
end
%�� ���� ����������� ������ ���������� � ��������� logData - ��� �����
%������ ������� ����� ������� � �� ������������ ����� �������� RT (� �� �
%������ �� � ��������, � � ������)
logData=analysAnagLog(logFile.name);
%���� ����� � ���������� fmriPrefix � AnatPrefix
fmriDir=dir(strcat(fmriPrefix,'*'));
anatDir=dir(strcat(AnatPrefix,'*'));
if length(fmriDir)~=1 || length(anatDir)~=1
    disp('Something wrong with fmri and/or anatomy directories');
end
%��������� ���������� ���� ���� ����������� ����� �� ������������� ����������
for i=1:length(maskFiles)
    maskFilesAbsPath{i}=strcat(subjectPath,'\',anatDir.name,'\rwrmask',maskFiles(i).name);
end
%��������� � ���� ���������� � ������ ��� ����� ���������� .nii � .img �
%��������� fmriFilePrefix.
cd(fmriDir.name);
fmriFiles=cat(1,dir(strcat(fmriFilePrefix,'*.nii')),dir(strcat(fmriFilePrefix,'*.img')));
if isempty(fmriFiles) || length(fmriFiles)<logData.time(end)
    disp('not enough fmri files');
end
%������ ��� ����� � ����� ���������� �������� ��� ������
for j=1:length(maskFiles)
    Imask=spm_read_vols(spm_vol(maskFilesAbsPath{j}));
    [coords(j).NZE1,coords(j).NZE2,coords(j).NZE3]=ind2sub(size(Imask),find(Imask));
end
f = waitbar(0,'Calculation in progress');
N=size(logData.time,2);
for b=1:4
    for i=1:N
        wwaitbarr=waitbar(((b-1)*N+i)/(4*N),f);
        curVec=zeros(1,length(maskFiles));
        %��� ������ ����� � ��� ������� ������� ������� ��������� ����
        %����� ������ ������������ (�� �������)� ��������� �� ��� ������
        %�������� � �������� �� ����������� �����. ����� �������� ���
        %������ ����� ���������. ��� ���� ��� ������� ����� � ������ RT
        %�������� ������������ � ����� ��� ������� �� dur (������������ �
        %������)
        for j=1:length(maskFiles)
            for k=1:logData.dur(b,i)
                V=spm_vol(fmriFiles(logData.time(i)+k).name);
                Ic=spm_sample_vol(V,coords(j).NZE1,coords(j).NZE2,coords(j).NZE3,1);
                %curVec(j)=curVec(j)+mean(Ic(~isnan(Ic)));
                curVec(j)=curVec(j)+ max(Ic(~isnan(Ic)))-min(Ic(~isnan(Ic)));
            end
            curVec(j)=curVec(j)/logData.dur(b,i);
        end
        vec=cat(1,vec,[curVec b]);
    end
    
end
close(wwaitbarr);
cd(Pathn);
maxminname='maxmin_'+string(DD(zz).name)+'.txt';
   save(maxminname, 'vec', '-ascii');
vec =[];
end


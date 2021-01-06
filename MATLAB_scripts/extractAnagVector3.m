function [vec] = extractAnagVector3(Pathn,fmriPrefix,AnatPrefix,fmriFilePrefix,maskDir,gray_matter_path)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%��������� � ����� � �������, ��������� ��� �������� ������ ���� .nii �
%.img
cd(maskDir);
maskFiles=cat(1,dir('*.nii'),dir('*.img'));
coords=struct;
vec_R=[];
vec_W=[];
TR = 3;


%��������� � ����� � ��������. ���������������, ��� � ��� �����
%���-����,������� ������ � ��� ��� ������������ � logFile
cd(Pathn);
%hdrf = [0 0.358019287921188 0.569864407163731 0.204148277822176 0.00239861063843975];
DD=dir('00*');
for zz=1:length(DD)
subjectPath = strcat(Pathn,DD(zz).name);
cd(subjectPath);
M = dlmread('RWT.txt');
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
    maskFilesAbsPath{i}=strcat(AnatPrefix,'\rwrmask',maskFiles(i).name);
end
%������ ��� ����� � ����� ���������� �������� ��� ������ 


Igraymatter = spm_read_vols(spm_vol(strcat(Pathn,DD(zz).name,'\',gray_matter_path,'\','rwrmaskgray_matter_wfu1.nii')));
for j=1:length(maskFiles)
    
    Imask=spm_read_vols(spm_vol(maskFilesAbsPath{j}));
    Imask = Imask .* Igraymatter;
    [coords(j).NZE1,coords(j).NZE2,coords(j).NZE3]=ind2sub(size(Imask),find(Imask));
    
end


%��������� � ���� ���������� � ������ ��� ����� ���������� .nii � .img �
%��������� fmriFilePrefix.
cd(fmriDir.name);
fmriFiles=cat(1,dir(strcat(fmriFilePrefix,'*.nii')),dir(strcat(fmriFilePrefix,'*.img')));
if isempty(fmriFiles) || length(fmriFiles)<logData.time(end)
    disp('not enough fmri files');
end
textwb =['Calculation in progress on ' DD(zz).name];
f = waitbar(0,textwb);
N=size(logData.time,2);
flagDur = false;
%for b=2:2
    for i=1:N
        wwaitbarr=waitbar(i/N,f);
        curVec=zeros(1,length(maskFiles));
        %��� ������ ����� � ��� ������� ������� ������� ��������� ����
        %����� ������ ������������ (�� �������)� ��������� �� ��� ������
        %�������� � �������� �� ����������� �����. ����� �������� ���
        %������ ����� ���������. ��� ���� ��� ������� ����� � ������ RT
        %�������� ������������ � ����� ��� ������� �� dur (������������ �
        %������)
        %if logData.dur(2,i)<4
           flagDur = true;
           for j=1:length(maskFiles)
               
                V=spm_vol(fmriFiles(logData.time(2,i)+logData.dur(2,i)).name);    
                Ic=spm_sample_vol(V,coords(j).NZE1,coords(j).NZE2,coords(j).NZE3,1);
            
                if sum(~isnan(Ic))>0
                                   
                    curVec(j)=mean(Ic(~isnan(Ic)));    %�������
                    %curVec(j)=curVec(j)+ hdrf(k)*(max(Ic(~isnan(Ic)))-min(Ic(~isnan(Ic))));
                    %curVec(j)=curVec(j)+std(hdrf(k)*Ic(~isnan(Ic)));
                else
                    curVec(j)=0;   
                end
           end
        %end
        if flagDur && (M(i,1) ~= 0)
            flagDur = false;
            if M(i,3)==1
                vec_R=cat(1,vec_R,curVec);
            else
                vec_W=cat(1,vec_W,curVec);
            end
        end
    end
    
%end

close(wwaitbarr);
cd(Pathn);
test_name='A-1_'+string(DD(zz).name)+'.txt';
   save('R'+test_name, 'vec_R', '-ascii');
   save('W'+test_name, 'vec_W', '-ascii');
   
vec_R = [];
vec_W = [];
end
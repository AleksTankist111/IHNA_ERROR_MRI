function [vec] = outputtxtdur(subjectPath)
vec=[];
%��������� � ����� � ��������. ���������������, ��� � ��� �����
%���-����,������� ������ � ��� ��� ������������ � logFile
cd(subjectPath);
logFile=dir('*.log');
if length(logFile)~=1
    disp('More than one log file');
end
%�� ���� ����������� ������ ���������� � ��������� logData - ��� �����
%������ ������� ����� ������� � �� ������������ ����� �������� RT (� �� �
%������ �� � ��������, � � ������)
logData=analysAnagLogSkakun1(logFile.name);
durarray1 = logData.dur;
save durarray.txt durarray1 -ascii
timearray1 = logData.time;
save timearray.txt timearray1 -ascii
end


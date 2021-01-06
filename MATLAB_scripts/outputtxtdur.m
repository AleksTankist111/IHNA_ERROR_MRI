function [vec] = outputtxtdur(subjectPath)
vec=[];
%переходим в папку с сессиями. подразумевается, что в ней лежит
%лог-файл,который ищется и его имя записывается в logFile
cd(subjectPath);
logFile=dir('*.log');
if length(logFile)~=1
    disp('More than one log file');
end
%из лога извлекается нужная информация в структуру logData - там будут
%лежать времена начал стимула и их длительность через параметр RT (и то и
%другое не в секундах, а в сканах)
logData=analysAnagLogSkakun1(logFile.name);
durarray1 = logData.dur;
save durarray.txt durarray1 -ascii
timearray1 = logData.time;
save timearray.txt timearray1 -ascii
end


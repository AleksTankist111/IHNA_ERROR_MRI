function [logData] = analysAnagLogSkakun1(logFile)
blocks={'1Cond' '2Cond' '3Cond' '4Cond'};
cond_dur=14.5;
TR=3;
file=fileread(logFile);
beg=regexp(file,'Paradigm Begin','start');
logData=struct;
for w=1:length(blocks)
    clear r;
    r=beg(length(beg))+regexp(file(beg(length(beg)):length(file)), blocks{w}, 'end');
    for k=1:length(r)
    logData.time(w,k)=str2double(file((r(k)+26):1:(r(k)+31)));
    if isnan(logData.time(w,k))
        logData.time(w,k)=0;
    end
    if k~=length(r)
        r1=regexp(file(r(k):r(k+1)),'RT','end');
    else
        r1=regexp(file(r(k):length(file)),'RT','end');        
    end
    if length(r1)>=2
    r2=regexp((file(r(k)+r1(1):(r1(2)+r(k)))),'Block','end');
    else
    r2=regexp(file((r(k)+r1(1)):length(file)),'Block','end');    
    end
    logData.dur(w,k)=(str2double(file((r(k)+r1(1)+1):(r(k)+r1(1)+r2(1)-6))))/1000;
    if logData.dur(w,k)==0
        logData.dur(w,k)=cond_dur;
    end
    end
end
end
function varargout = interf_viewer_test2(varargin)
% INTERF_VIEWER_TEST2 MATLAB code for interf_viewer_test2.fig
%      INTERF_VIEWER_TEST2, by itself, creates a new INTERF_VIEWER_TEST2 or raises the existing
%      singleton*.
%
%      H = INTERF_VIEWER_TEST2 returns the handle to a new INTERF_VIEWER_TEST2 or the handle to
%      the existing singleton*.
%
%      INTERF_VIEWER_TEST2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in INTERF_VIEWER_TEST2.M with the given input arguments.
%
%      INTERF_VIEWER_TEST2('Property','Value',...) creates a new INTERF_VIEWER_TEST2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before interf_viewer_test2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to interf_viewer_test2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help interf_viewer_test2

% Last Modified by GUIDE v2.5 01-Sep-2018 20:43:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @interf_viewer_test2_OpeningFcn, ...
                   'gui_OutputFcn',  @interf_viewer_test2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before interf_viewer_test2 is made visible.
function interf_viewer_test2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to interf_viewer_test2 (see VARARGIN)

% Choose default command line output for interf_viewer_test2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


% UIWAIT makes interf_viewer_test2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = interf_viewer_test2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
choosesettings(handles)






% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3


% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox4


% --- Executes on button press in checkbox5.
function checkbox5_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox5




function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
choosesettings(handles);


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
opendirectory(handles);

%Открываем нужную папку и добавляем из нее в выпадающий список все файлы с
%данными + еще один (объединенный)
function opendirectory(handles)
subjectPath=get(handles.edit2,'string');
cd(subjectPath);
usabfiles=dir('*_*.txt');
p=length(usabfiles);
nar1=cell(1,p+1);
ma_files=cell(p+1);
nar1{1}=usabfiles(1).name;
ma_files{1}=dlmread(usabfiles(1).name);
ma_files{p+1}=ma_files{1};
    for j=2:p
        nar1{j}=usabfiles(j).name;
        ma_files{j}=dlmread(usabfiles(j).name);
        ma_files{p+1} =cat(1,ma_files{p+1},ma_files{j});
    end
 savematall=ma_files{p+1};
 save All.txt savematall -ascii
 nar1{p+1}='All.txt';  
 namesarray=cellstr(nar1);
 set(handles.popupmenu1,'string',namesarray); 
 
 
 function choosesettings(handles)
 ppp=0;
 maskpath1=get(handles.edit3,'string');
  %if (maskpath1 ~= '') && (maskpath1 ~= 'Директория и имя файла txt')
   % fid=fopen(get(handles.edit3,'string'));
    %masknames=textscan(fid,'%s %s');
    %fclose(fid);
    %ppp=1;
  %end
 contents = cellstr(get(handles.popupmenu1,'String'));
 tarr=dlmread(contents{get(handles.popupmenu1,'Value')});
 w=size(tarr,2);
 h=size(tarr,1);
 clarr= [get(handles.checkbox2,'Value') get(handles.checkbox3,'Value') 
     get(handles.checkbox4,'Value') get(handles.checkbox5,'Value')];
 points=1;
 masknum=str2double(get(handles.edit1,'String'));
 cla(handles.axes1);
 
 for i=1:h
    if clarr(tarr(i,w))== 1
        if tarr(i,w)==1
            colorp='bo';
        elseif tarr(i,w)== 2
            colorp='go';
        elseif tarr(i,w)== 3
            colorp='ro';
        else colorp='yo';
        end
        plot(points,tarr(i,masknum),colorp);
        hold on;
        points= points+1;
        if points>30 
           points=1; 
        end
    else 
        points = 1;
    end   
 end
 titlenum=' mask '+string(masknum)+'/'+string(w-1);
 if ppp==1
 title(masknames{2}{masknum}+titlenum);
 else title(titlenum);
 end
 
     
 


    



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

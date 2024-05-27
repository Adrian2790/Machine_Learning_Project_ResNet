function [XTrain, YTrain, XTest,YTest,XVal, YVal] = buildDatasets(X,Y,pTest,pVal)
% separate the dataset [X, Y] into training, validation, testing
% if targets are not available, use Y=[];


if nargin < 3 %default values for ptest and pValue
    pTest = 0.2;
    pVal=0;
end

N=size(X,1); % the number of all available samples
partition_N= cvpartition(N,'Holdout',pTest); %a partition with pTest*N samples for testing 

%% training 
idxTrain=training(partition_N); %the indices of the training samples 
XTrain=X(idxTrain,:); %the training samples - input patterns
if ~isempty(Y)
    YTrain=Y(idxTrain,:);%the training samples - targets
else
    YTrain=[];
end

%% testing 
idxTest=~idxTrain; %the indices of the testing samples 
XTest=X(idxTest,:); %the testing samples - input patterns
if ~isempty(Y)
    YTest=Y(idxTest,:);%the testing samples - targets
else
    YTest=[];
end

%if validation dataset if requested
if pVal>0
    Nt=size(XTrain,1); % the number of training samples
    partition_Nt= cvpartition(Nt,'Holdout',pVal/(1-pTest)); %pVal*N = pVal/(1-pTest)* Nt samples for validation 
    
      
    idxTrain=training(partition_Nt); % the indices of the remaining training samples 

    %% validation dataset
    idxVal=~idxTrain;  
    XVal=XTrain(idxVal,:);
    if ~isempty(Y)
        YVal=YTrain(idxVal,:);
    else
        YVal=[];
    end

    %% training dataset - reconfigured
    XTrain=XTrain(idxTrain,:); 
    if ~isempty(Y)
        YTrain=Y(idxTrain,:);
    else
        YTrain=[];
    end
else
    XVal=[];
    YVal=[];
end

end

%%"C:\Users\sofia\Desktop\IS_TRAFFIC\traffic_signs_p"

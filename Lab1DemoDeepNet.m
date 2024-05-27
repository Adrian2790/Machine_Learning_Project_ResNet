%% CNN for image classification
%+++++++++++++++++++++++++++++++++++
 
clear all
close all
clc

%% PARAMETERS
% the name of the file where the trained CNN is saved
nameFileRez='rezCNN.mat';

% training parameters
MBS=80; % mini batch size
NEP=20;  % number of epochs

% indicate the path to images
pathImagesTrain='C:\Users\sofia\Desktop\IS_TRAFFIC\traffic_sign_train';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize

pathImagesTest='C:\Users\sofia\Desktop\IS_TRAFFIC\traffic_sign_test';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize

%% PRE_TRAINED MODEL

% load the pretrained model 
[net, classes] = imagePretrainedNetwork("resnet18");
% net = vgg16;

% the size of the input images
inputSize = net.Layers(1).InputSize;

%% TRAINING AND VALIDATION DATASETS

% create the datastore with the training and validation images
imds = imageDatastore(pathImagesTrain, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

% resize the input images, if necessary
im=readimage(imds,1);
if size(im,1)~=inputSize(1) || size(im,2)~=inputSize(2)
    resizeImagesImdsN(imds, pathImagesTrain, inputSize(1:2), 'C:\Users\sofia\Desktop\IS_TRAFFIC\resizedData');
    imds = imageDatastore('C:\Users\sofia\Desktop\IS_TRAFFIC\resizedData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

% split the dataset into training and validation datasets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% obtain information about the training dataset
numTrainImages = numel(imdsTrain); % the number of trainig images 
numClasses = numel(categories(imdsTrain.Labels)); % the number of classes

% augment the training and validation dataset
pixelRange = [-30 30];   
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


%% TESTING DATASETS

% create the datastore for the testing dataset      
imdsTest = imageDatastore(pathImagesTest, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');   

% resize the input images, if necessary
im=readimage(imdsTest,1);
if size(im,1)~=inputSize(1) || size(im,2)~=inputSize(2)
    resizeImagesImdsN(imdsTest, pathImagesTest, inputSize(1:2), 'C:\Users\sofia\Desktop\IS_TRAFFIC\resizedTest');
    imdsTest = imageDatastore('C:\Users\sofia\Desktop\IS_TRAFFIC\resizedTest', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

%% DESIGN THE ARCHITECTURE

% take the layers for transfer of learning
gNet=layerGraph(net);

% create the new architecture: the last fully connected layer is configured for the necessary number of classes
newLayer1 = fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,"Name",net.Layers(end-2).Name); 
newLayer2 = softmaxLayer("Name",net.Layers(end-1).Name);
newLayer3 = classificationLayer("Name",net.Layers(end).Name);

gNet=replaceLayer(gNet,net.Layers(end-2).Name,newLayer1);
gNet=replaceLayer(gNet,net.Layers(end-1).Name,newLayer2);
gNet=replaceLayer(gNet,net.Layers(end).Name,newLayer3);

% analyzeNetwork(gNet)

%% TRAIN THE CNN

% indicate the training parameters
options = trainingOptions('sgdm', ...
    'MiniBatchSize',MBS, ...            
    'MaxEpochs',NEP, ...      
    'InitialLearnRate',1e-4, ...  
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
                  
% train the model
netTransfer = trainNetwork(imdsTrain,gNet,options);

% save the trained model
feval(@save,nameFileRez,'netTransfer'); 

%% VERIFY THE RESULTS 

% validation - responses and accuracy
[YPredValidation,scoresValidation] = classify(netTransfer,imdsValidation); 
accuracyValidation = mean(YPredValidation == imdsValidation.Labels);  

% training - responses and accuracy
[YPredTrain,scoresTrain] = classify(netTransfer,imdsTrain);  
accuracyTrain = mean(YPredTrain == imdsTrain.Labels);  

% testing- responses and accuracy
[YPredTest,scoresTest] = classify(netTransfer,imdsTest);  
accuracyTest = mean(YPredTest == imdsTest.Labels);  

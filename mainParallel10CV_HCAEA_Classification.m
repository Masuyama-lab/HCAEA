% 
% (c) 2022 Naoki Masuyama
% 
% These are the codes of CIM-based Adaptive Resonance Theory with Age and Edge (CAEA) and Hierarchical CAEA (HCAEA)
% proposed in N. Masuyama, N. Amako, Y. Yamada, Y. Nojima, and H. Ishibuchi, 
% "Adaptive resonance theory-based clustering with a divisive hierarchical structure capable of continual learning,"
% IEEE Access, 2022."
% 
% Please contact "masuyama@omu.ac.jp" if you have any problems.
%    

clear all
close all

% Experimental Conditions =================================================
numModel = 1; % Number of comparison models
Loop = 2;  % Number of Loops for averaging
kfold = 10;     % kfold-cross validation
Epoch = 1; % Number of epochs for data input count

% maxlevel = 1: CAEA, maxlevel >= 2: HCAEA 
maxlevel = 10;

isSave = true;

% Datasets
data_list = [{'iris'},{'Wine'}];
% data_list = [{'iris'}];


param1 = 10; % \ageMax1
param2 = 28; % \lambda
%  ========================================================================

h = waitbar(0, 'Waiting for FevalFutures to complete...');

parfeval_job(1:Loop*kfold) = parallel.FevalFuture;
for dataIdx = 1:size(data_list,2)

    % Save Directory
    dir = strcat('.../Result_classification/');
    dir_name = strcat(dir,char(data_list(dataIdx)));
    mkdir(dir_name)

    i = 1;


    disp(strcat('ageMax = ', num2str(param1), ', Lambda = ', num2str(param2), ', ', data_list(dataIdx)));

    % Setting up jobs
    for cvIdx = 1:Loop*kfold
        % parfeval(job, @XXXX, #outputsOf@XXXX, x1,x2,...xn)
        parfeval_job(cvIdx) = parfeval(@Parallel_10CV_HCAEA_Classification, 2, char(data_list(dataIdx)), numModel, cvIdx, Epoch, kfold, param1, param2, maxlevel);
    end

    for cvIdx = 1:Loop*kfold
        % Output of parfeval_job
        [completedIdx, Record, HCAEAnet] = fetchNext(parfeval_job);

        Record_ALL.Acc(completedIdx,i) = Record.Acc;
        Record_ALL.NMI(completedIdx,i) = Record.NMI;
        Record_ALL.ARI(completedIdx,i) = Record.ARI;
        Record_ALL.MicroFS(completedIdx,i) = Record.MicroFS;
        Record_ALL.MacroFS(completedIdx,i) = Record.MacroFS;
        Record_ALL.NumLeaveNode(completedIdx,i) = Record.NumLeaveNode;
        Record_ALL.NumNode(completedIdx,i) = Record.NumNode;
        Record_ALL.MaxLevel(completedIdx,i) = Record.MaxLevel;
        Record_ALL.Noc(completedIdx,i) = Record.Noc;
        Record_ALL.PTime(completedIdx,i) = Record.PTime;

    end

    i = i+1;

    % Save Output
    if isSave == true
        save(strcat(dir_name,'/ageMax_',num2str(param1),'_lambda_',num2str(param2),'.mat'),strcat('Record_ALL'));
    end
    waitbar((dataIdx-1)/size(data_list,2),h,sprintf('Finished: Data %d/%d, ageMax %d, Lambda %d',dataIdx-1,size(data_list,2),param1, param2));


    disp('Finished Data: '); disp(data_list(dataIdx));

    % Accuracy
    mean(Record_ALL.Acc)
    % NMI
    mean(Record_ALL.NMI)
end

delete(h)




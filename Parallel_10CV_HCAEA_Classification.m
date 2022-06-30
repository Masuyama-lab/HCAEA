function [Record, HCAEAnet]  = Parallel_10CV_HCAEA_Classification(data_name, numModel, cvIdx, Epoch, kfold, param1, param2, maxlevel)

% Parallel_10CV_HCAEA_Classification('wine', 1, 1, 1, 10, 10, 6, 3);

rng(cvIdx*Epoch);

% convert cvIdx to fold idx
fold = mod(cvIdx,10);
fold(mod(cvIdx,10) == 0) = 10;

tmpData = load(strcat(data_name,'.mat'));
IMAGES = tmpData.data;
LABELS = tmpData.target;

%% Data preparation ----------------------------------------------------

if size(LABELS,2)>1
    tmplabel = rem(find(LABELS'>=1), size(LABELS,2));
    tmplabel(tmplabel==0) = size(LABELS,2);
    LABELS = tmplabel;
end

% % Z-Score
% %IMAGES = normalize(IMAGES,'range');
% IMAGES = normalize(IMAGES);

% Randamization
ran = randperm(size(IMAGES,1));
IMAGES = IMAGES(ran,:);
LABELS = LABELS(ran);


% For k-fold
% Accuracy preservation
cvAcc = zeros(1,numModel);
% Number of Leaf Nodes
cvNumLeaveNode = zeros(1,numModel);
% Number of Nodes preservation
cvNumNode = zeros(1,numModel);
% Number of Layer
cvMaxLevel = zeros(1,numModel);
% Number of Clusters preservation
cvNoc = zeros(1,numModel);
% Processing Time preservation
cvPTime = zeros(1,numModel);
% Normalized Mutual Information
cvNMI = zeros(1,numModel);
% micro F-score
cvMicroFS = zeros(1,numModel);
% macro F-score
cvMacroFS = zeros(1,numModel);
% Adjusted Rand Index
cvARI = zeros(1,numModel);

% For kfold*Loop
% Accuracy preservation
Record.Acc = zeros(1,numModel);
% Number of Leaf Nodes
Record.NumLeaveNode = zeros(1,numModel);
% Number of Nodes preservation
Record.NumNode = zeros(1,numModel);
% Number of Layer
Record.MaxLevel = zeros(1,numModel);
% Number of Clusters preservation
Record.Noc = zeros(1,numModel);
% Processing Time preservation
Record.PTime = zeros(1,numModel);
% Normalized Mutual Information
Record.NMI = zeros(1,numModel);
% micro F-score
Record.MicroFS = zeros(1,numModel);
% macro F-score
Record.MacroFS = zeros(1,numModel);
% Adjusted Rand Index
Record.ARI = zeros(1,numModel);



%%

% data partitioning
cv = cvpartition(size(IMAGES,1),'kfold',kfold);

% Set an index for fold
itrCV = fold;

% parfor itrCV = 1:kfold %parfor

    %Index extraction for cross validation
    trainIdx = training(cv, itrCV);
    testIdx = test(cv, itrCV);
    trainData = IMAGES(trainIdx,:);
    trainLabels = LABELS(trainIdx,1);
    testData = IMAGES(testIdx,:);
    testLabels = LABELS(testIdx,1);

    maxLABEL = max(LABELS);

    %   Parameters of HCAEA ===================================================
    HCAEAnet.numNodes    = 0;   % the number of nodes
    HCAEAnet.weight      = [];  % node position
    HCAEAnet.CountNode = [];    % winner counter for each node
    HCAEAnet.adaptiveSig = [];  % kernel bandwidth for CIM in each node
    HCAEAnet.edge = [];         % Initial connections (edges) matrix
    HCAEAnet.edgeAge = [];      % Age of edge
    HCAEAnet.LabelCluster = [];
    HCAEAnet.CountEdge = [];
    HCAEAnet.CIMthreshold = [];

    HCAEAnet.Epochs=1;
    HCAEAnet.MaxLevel = maxlevel;

    HCAEAnet.Lambda = param2;       % an interval for calculating a kernel bandwidth for CIM
    HCAEAnet.edgeAgeMax = param1;       % Maximum node age

    HCAEAnet.CountLabel = [];
    % ====================================================================

    disp([HCAEAnet.Lambda,HCAEAnet.edgeAgeMax]);


    time_HCAEA = 0;


    % create data
    ranData = [];
    ranLabels = [];

    for nitr = 1:Epoch

        % Randamization
        ran = randperm(size(trainData,1));
        ranD = trainData(ran,:);
        ranL = trainLabels(ran);

        ranData = [ranData ranD'];
        ranLabels = [ranLabels ranL'];

    end 
    ranLabels = ranLabels';


    % HCAEA =============================================
    Level = 1;

    tic
    [HCAEAnet,TF] = TrainHCAEA_Classification(ranData,HCAEAnet,Level,ranLabels,maxLABEL);
    time_HCAEA = time_HCAEA + toc/(Epoch*size(trainData,1));
    
    
    
    if TF == 0
        if ~(HCAEAnet.numNodes == 0)
            %==================================================================
            %get leaves
            [LEAVESnet,MaxLevel] = GetLEAVESnet_Classification(HCAEAnet,0,0,0,0,0);
            LEAVESnet.weight = LEAVESnet.Means;
            LEAVESnet.CountLabel = LEAVESnet.CL;
            %==================================================================

            % Evaluation ------------------------------------------------------
            [HCAEA_ACC, HCAEA_normMI, HCAEA_microF, HCAEA_macroF, HCAEA_ARI] = HCAEA_Evaluation_C(testData, testLabels, LEAVESnet);
            num_nodes_HCAEA = CountNumNodes(HCAEAnet);


            % Accuracy
            cvAcc(1,:) = HCAEA_ACC;
            % Normalized Mutual Information
            cvNMI(1,:) = HCAEA_normMI;
            % micro F-score
            cvMicroFS(1,:) = HCAEA_microF;
            % macro F-score
            cvMacroFS(1,:) = HCAEA_macroF;
            % Adjusted Rand Index
            cvARI(1,:) = HCAEA_ARI;
            % Number of Leaves
            cvNumLeaveNode(1,:) = LEAVESnet.numNodes;
            % Number of Nodes
            cvNumNode(1,:) = num_nodes_HCAEA;
            % Number of Layer
            cvMaxLevel(1,:) = MaxLevel;
            % Number of Clusters
            cvNoc(1,:) = max(LEAVESnet.LabelCluster);
            % Processing Time
            cvPTime(1,:) = time_HCAEA;
        end
    end

% end % itrCV = 1:kfold


    % Accuracy
    Record.Acc(1,:) = cvAcc;
    % Normalized Mutual Information
    Record.NMI(1,:) = cvNMI;
    % micro F-score
    Record.MicroFS(1,:) = cvMicroFS;
    % macro F-score
    Record.MacroFS(1,:) = cvMacroFS;
    % Adjusted Rand Index
    Record.ARI(1,:) = cvARI;
    % Number of Leaves
    Record.NumLeaveNode(1,:) = cvNumLeaveNode;
    % Number of Nodes
    Record.NumNode(1,:) = cvNumNode;
    % Number of Layer
    Record.MaxLevel(1,:) = cvMaxLevel;
    % Number of Clusters
    Record.Noc(1,:) = cvNoc;
    % Processing Time
    Record.PTime(1,:) = cvPTime;


end





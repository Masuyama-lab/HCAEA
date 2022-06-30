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
function Model = TrainCAEA_Classification(Samples,net,Level,NumSteps,SampleLabels,maxLABEL)

Model.Level = Level;
Model.Samples=Samples;
Model.SampleLabels = SampleLabels;
DATA=Samples.'; 
Model.NumSteps=NumSteps; % Total number of steps
Model.Winners=zeros(1,size(Samples,2));

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
Lambda = net.Lambda;             % an interval for calculating a kernel bandwidth for CIM

edge = net.edge;
edgeAge = net.edgeAge;
edgeAgeMax = net.edgeAgeMax;

CountEdge = net.CountEdge; 
CIMthreshold = net.CIMthreshold;

CountLabel = net.CountLabel;

% Set a size of CountLabel
if size(weight) == 0
    CountLabel = zeros(1, maxLABEL);
end


for sampleNum = 1:NumSteps
    
    index = mod(sampleNum, size(DATA, 1));
    if index == 0
        index = size(DATA, 1); 
    end
    
    % Current data sample.
    input = DATA(index,:);
    label = SampleLabels(index, 1);
    
    % The number of inputs that directly becomes nodes.
    bufferInput = round(Lambda/2);
    
    if size(weight,1) < bufferInput % In the case of the number of nodes in the entire space is small.
        % Add Node
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        CountNode(numNodes) = 1;
        adaptiveSig(numNodes) = SigmaEstimation(DATA, sampleNum, bufferInput);
        edge(numNodes, :) = 0;
        edge(:, numNodes) = 0;
        edgeAge(numNodes, :) = 0;
        edgeAge(:, numNodes) = 0;
        CountEdge(numNodes, :) = 0;
        CountEdge(:, numNodes) = 0;
        
        Model.Winners(sampleNum)=numNodes;
        CountLabel(numNodes,label) = 1;
        
        % Assign similarlity threshold to the initial nodes.
        if numNodes == bufferInput
            tmpTh = zeros(1,bufferInput);
            for k = 1:bufferInput
                tmpCIMs1 = CIM(weight(k,:), weight, mean(adaptiveSig));
                [~, s1] = min(tmpCIMs1);
                tmpCIMs1(s1) = inf; % Remove CIM between weight(k,:) and weight(k,:). 
                tmpTh(k) = min(tmpCIMs1); 
            end
            CIMthreshold = repmat(mean(tmpTh), 1, bufferInput); 
            
        else
            CIMthreshold(1:numNodes) = mean(CIMthreshold);
        end
        
    else
        
        % Calculate CIM based on global mean adaptiveSig.
        globalCIM = CIM(input, weight, mean(adaptiveSig));
        gCIM = globalCIM;
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Vs1, s1] = min(gCIM);
        gCIM(s1) = inf;
        [Vs2, s2] = min(gCIM);
        
        if CIMthreshold(s1) < Vs1 % Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes) = SigmaEstimation(DATA, sampleNum, bufferInput);
            edge(numNodes, :) = 0;
            edge(:, numNodes) = 0;
            edgeAge(numNodes, :) = 0;
            edgeAge(:, numNodes) = 0;
            CountEdge(numNodes, :) = 0;
            CountEdge(:, numNodes) = 0;
            
            Model.Winners(sampleNum)=numNodes;
            CountLabel(numNodes,label) = 1;

            
            % Assigne similarlity threshold
            CIMthreshold(numNodes) = CIMthreshold(s1);
%             s1Neighbors = find( edge(s1,:) );
%             if isempty(s1Neighbors) == 1 
%                 CIMthreshold(numNodes) = CIMthreshold(s1);
%             else 
%                 tmpCIMs1 = CIM(weight(s1,:), weight(s1Neighbors,:), mean(adaptiveSig(s1Neighbors))); 
%                 s1_s1NeighborCIM = [tmpCIMs1, CIMthreshold(s1)];
%                 CIMthreshold(numNodes) = mean(s1_s1NeighborCIM);
%             end
            
        else % Case 2 i.e., V >= Vs1
            
            % Increment age
            edgeAge(s1,:) = edgeAge(s1,:) + 1;
            edgeAge(:,s1) = edgeAge(:,s1) + 1;
            
            % If age > ageMAX, detele edge
            deleteAge = find( edgeAge(s1,:) > edgeAgeMax );
            edge(s1, deleteAge) = 0;
            edge(deleteAge, s1) = 0;
            edgeAge(s1, deleteAge) = 0;
            edgeAge(deleteAge, s1) = 0;
            CountEdge(s1, deleteAge) = 0;
            CountEdge(deleteAge, s1) = 0;
            
            % Update s1 weight
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            CountNode(s1) = CountNode(s1) + 1;
            CountEdge(s1,s2) = CountEdge(s1,s2) + 1; %???
            
            Model.Winners(sampleNum)=s1;
            CountLabel(s1,label) = CountLabel(s1, label) + 1;
            
            % Update s1 neighbor
            if CIMthreshold(s2) >= Vs2 % Case 3 i.e., V >= CIM_k2
                % Update weight of s2 node.
                s1Neighbors = find( edge(s1,:) );
                for k = s1Neighbors
                    weight(k,:) = weight(k,:) + ( 1/(10*CountNode(k) )) * (input - weight(k,:));
                end
                
                % Create an edge between s1 and s2 nodes.
                edge(s1,s2) = 1;
                edge(s2,s1) = 1;
                edgeAge(s1,s2) = 0;
                edgeAge(s2,s1) = 0; 
            end
            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
    
    % Topology Adjustment
    % If activate the following function, CAEAF shows a noise reduction ability.
    if mod(sampleNum, Lambda) == 0 && size(weight,1) > 1
        % -----------------------------------------------------------------
        % Delete Node based on number of neighbors
        nNeighbor = sum(edge);
        deleteNodeEdge = (nNeighbor == 0);
        
        % Delete process
        numNodes = numNodes - sum(deleteNodeEdge);
        weight(deleteNodeEdge, :) = [];
        CountNode(deleteNodeEdge) = [];
        edge(deleteNodeEdge, :) = [];
        edge(:, deleteNodeEdge) = [];
        edgeAge(deleteNodeEdge, :) = [];
        edgeAge(:, deleteNodeEdge) = [];
        adaptiveSig(deleteNodeEdge) = [];
        CIMthreshold(deleteNodeEdge) = [];
        CountLabel(deleteNodeEdge, :) = [];
        
        deleteIdx = find(deleteNodeEdge);
        for i = 1:size(deleteIdx,2) 
            Model.Winners(Model.Winners==deleteIdx(i)) = -1; 
            Model.Winners(Model.Winners>deleteIdx(i)) = Model.Winners(Model.Winners>deleteIdx(i))-1; 
        end
        
    end % if mod(sampleNum, Lambda) == 0
    
    
    % Drawing
%     if mod(sampleNum, 500) == 0
%         net.weight = weight;
%         net.edge = edge;
%         connection = graph(edge ~= 0);
%         net.LabelCluster = conncomp(connection);
%         try
%             set(0,'CurrentFigure',2)
%         catch
%             figure(2);
%         end
%         myPlot(DATA, net, 'CAEAF');
%         drawnow
%     end
    
    
    
end % for sampleNum = 1:size(DATA,1)

% %reallocate
% [winners,CountLabel] = reallocate(weight, adaptiveSig, CIMthreshold, Model.Winners, CountLabel, DATA, SampleLabels);
% Model.Winners = winners;

% Cluster Labeling based on edge (Functions are available above R2015b.)
connection = graph(edge ~= 0);
LabelCluster = conncomp(connection);

Model.numNodes = numNodes;      % Number of nodes
Model.weight = weight;          % Mean of nodes
Model.CountNode = CountNode;    % Counter for each node
Model.adaptiveSig = adaptiveSig;

Model.LabelCluster = LabelCluster;
Model.edge = edge;
Model.edgeAge = edgeAge;
Model.CountEdge = CountEdge;
Model.CIMthreshold = CIMthreshold;

Model.CountLabel = CountLabel;
Model.CL = CountLabel';

end


% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimation(DATA, sampleNum, bufferInput)

if (sampleNum - bufferInput) <= 0
    exNodes = DATA(1:sampleNum,:);
elseif (sampleNum - bufferInput) > 0
    exNodes = DATA( (sampleNum+1)-bufferInput:sampleNum, :);
end

% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[n, att] = size(Y);
g_Kernel = zeros(n, att);

for i = 1:att
    g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
end

% ret0 = GaussKernel(0, sig);
ret0 = 1;
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end

%reallocate
function [winners,CountLabel] = reallocate(weight, adaptiveSig, CIMthreshold, winners, CountLabel, DATA, SampleLabels)

Candidates = find(winners==-1);
CandidatesSize = size(Candidates);

for i = 1:CandidatesSize(2)
    index = Candidates(i);
    input = DATA(index,:);
    label = SampleLabels(index);
    
    % Calculate CIM based on global mean adaptiveSig.
    globalCIM = CIM(input, weight, mean(adaptiveSig));
    gCIM = globalCIM;

    % Set CIM state between the local winner nodes and the input for Vigilance Test.
    [Vs1, s1] = min(gCIM);

    if CIMthreshold(s1) >= Vs1 % Case 2, 3
        % reallocate to s1
        winners(index)=s1;
        CountLabel(s1,label) = CountLabel(s1, label) + 1;
    end
end
end




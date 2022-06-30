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
function [Model, TF] = TrainHCAEA_Classification(Samples,net,Level,SampleLabels,maxLABEL)

% TRAINFTCA  Create the CAEA tree.

%%
TF = 0;
Model = [];
MaxLevel = net.MaxLevel;
[Dimension,NumSamples]=size(Samples);
if ((NumSamples<(Dimension+1)) && (Level>1)) || (Level>MaxLevel)    
    return;
end

%fprintf('\nLEVEL=%d\n',Level);

%% Growing Process
NumSteps = net.Epochs*NumSamples;
Model = TrainCAEA_Classification(Samples,net,Level,NumSteps,SampleLabels,maxLABEL);

%% Expansion Process
Winners = Model.Winners;
Model.Means = Model.weight.';
NeuronsIndex = find(isfinite(Model.Means(1,:)));
NumNeurons = numel(NeuronsIndex);

%fprintf('Final Graph Neurons: %d\n',NumNeurons);

%%
Model.Connections = sparse(Model.edge);

%% PRUNE THE GRAPHS WITH ONLY 2 NEURONS. THIS IS TO SIMPLIFY THE HIERARCHY
if NumNeurons==2
    Model=[];
    TF = 1;
    return;
else
    for NeuronIndex=NeuronsIndex
        ChildSamples = Samples(:,Winners==NeuronIndex);
        ChildSampleLabels = SampleLabels(Winners==NeuronIndex);
        Model.Child{NeuronIndex} = TrainHCAEA_Classification(ChildSamples,net,Level+1,ChildSampleLabels,maxLABEL);
    end
end

end


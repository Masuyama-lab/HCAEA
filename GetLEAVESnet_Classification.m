function [LEAVESnet,MaxLevel] = GetLEAVESnet_Classification(HFTCAnet,M,C,L,A,MaxLevel)

NewMeans = [];
NewCL = [];
NewLabelCluster = [];
NewadaptiveSig = [];

if HFTCAnet.numNodes == 0
    NewMeans = [NewMeans M];
    NewCL = [NewCL C];
    NewLabelCluster = [NewLabelCluster L];
    NewadaptiveSig = [NewadaptiveSig A];
    if HFTCAnet.Level > MaxLevel
        MaxLevel = HFTCAnet.Level;
    end
else
    IndexValidNeurons = find(isfinite(HFTCAnet.Means(1,:)));

    for NeuronIndex=IndexValidNeurons
        if ~isempty(HFTCAnet.Child{NeuronIndex})
            M = HFTCAnet.Means(:,NeuronIndex);
            C = HFTCAnet.CL(:,NeuronIndex);
            L = HFTCAnet.LabelCluster(:,NeuronIndex);
            A = HFTCAnet.adaptiveSig(:,NeuronIndex);
            if HFTCAnet.Level > MaxLevel
                MaxLevel = HFTCAnet.Level;
            end
            [LEAVESnet_Child,MaxLevel] = GetLEAVESnet_Classification(HFTCAnet.Child{NeuronIndex},M,C,L,A,MaxLevel); 
            NewMeans = [NewMeans LEAVESnet_Child.Means];
            NewCL = [NewCL LEAVESnet_Child.CL];
            NewLabelCluster = [NewLabelCluster LEAVESnet_Child.LabelCluster];
            NewadaptiveSig = [NewadaptiveSig LEAVESnet_Child.adaptiveSig];
            
        else
            NewMeans = [NewMeans HFTCAnet.Means(:,NeuronIndex)];
            NewCL = [NewCL HFTCAnet.CL(:,NeuronIndex)];
            NewLabelCluster = [NewLabelCluster HFTCAnet.LabelCluster(:,NeuronIndex)];
            NewadaptiveSig = [NewadaptiveSig HFTCAnet.adaptiveSig(:,NeuronIndex)];
            if HFTCAnet.Level > MaxLevel
                MaxLevel = HFTCAnet.Level;
            end
        end    
    end
end

LEAVESnet.Means = NewMeans;
LEAVESnet.CL = NewCL;
LEAVESnet.LabelCluster = NewLabelCluster;
LEAVESnet.adaptiveSig = NewadaptiveSig;
LEAVESnet.numNodes = size(NewMeans,2);




function num_nodes_hftca = CountNumNodes(net)
        k = 1;
        net.GraphIndex = k;
        queue{1} = net;
        
        label(k) = size(net.Samples,2);

        
       while (~isempty(queue))
           currentModel = queue{1};
           if (length(queue) > 1)
               queue(1) = []; %pop neuron
           else
               queue = []; %pop neuron
           end
           
           NdxValidNeurons = find(isfinite(currentModel.Means(1,:)));
           fatherIndex = currentModel.GraphIndex;
           
           
        for NdxNeuro = NdxValidNeurons        
            childIndex = k + 1;
             if ~isempty(currentModel.Child{NdxNeuro})
                 currentModel.Child{NdxNeuro}.GraphIndex = childIndex;
                 label(childIndex) = size(currentModel.Child{NdxNeuro}.Samples,2); 
                 if (~isempty(queue))
                    queue = [queue, currentModel.Child(NdxNeuro)];
                 else
                    queue = [currentModel.Child(NdxNeuro)];
                 end
             else
                 label(childIndex) = sum(currentModel.Winners == NdxNeuro);
             end
             
                s(k) = fatherIndex;
                t(k) = childIndex;
                k = k + 1;
                
                
        end
       
       end
  
        s = s(~isnan(s));    
        t = t(~isnan(t));
        
        [~, num_nodes_hftca] = size(t);
end

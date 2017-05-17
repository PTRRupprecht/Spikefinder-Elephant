%% Uses parameters from 'MetaParameterExtraction.m' and 'SingleNeuronCNNs.py' to generate embedding spaces via PCA
% This was done in Matlab mainly for plotting.

load('Confusion.mat');

% normalize values for each neuron by how well this neuron be predicted by others (maximum except self-prediction)
ConfusionX = Confusion_later;
for j = 1:174
    ConfusionX(:,j) = ConfusionX(:,j)/mean(ConfusionX(:,j));
end


%% calculate predicting quality

% normalize values for each neuron by how well this neuron can be predicted (maximum except self-prediction)
CX = Confusion_later - eye(size(Confusion_later)).*Confusion_later;
for k = 1:size(CX,1)
    CX(k,:) = CX(k,:)/quantile(CX(k,:),0.98);
end

% find for each cell k the neurons that can be predicted very well;
% rank neurons according to how well they predict neurons compared to
% how well those neurons are predicted by other well neurons
clear predicting_quality
for k = 1:size(CX,1)
    for j = 1:size(CX,1)
        [XX,iX] = sort(CX(j,:),'descend');
        position(j) = find(iX == k);
    end
    position_sorted = sort(position,'ascend');
    predicting_quality(k) = 1/mean(position_sorted(1:10));
end
% figure(32), plot(predicting_quality,'-')

%% select neurons to be used for PCA

goodindizes = find(predicting_quality>quantile(predicting_quality,0.05));
DaSet2 = DaSet2(goodindizes);
DaSet = DaSet(goodindizes);


%% perform PCA to embed the prediction matrix into a lower dimension


ConfusionXX = ConfusionX(goodindizes,goodindizes);
for j = 1:10
    for k = 1:10
        indizes1 = find(DaSet==j);
        indizes2 = find(DaSet==k);
        ConfusionXX(indizes1,indizes2) = mean(mean(ConfusionXX(indizes1,indizes2)));
    end
end
figure, imagesc((ConfusionXX+ConfusionXX')/2,[0.5 2.5])
figure, imagesc(ConfusionX(goodindizes,goodindizes),[0.5 2.5])
[~,mappedX,~] = pca(ConfusionXX'+ConfusionXX); 


symbolsX = {'o','x','s','d','^','p','h','*','+','>'};
numbersX = {'1','2','3','4','5','6','7','8','9','10'};

figure(95533), cmap = jet(10);
for k = 1:10
    indizes = find(DaSet==k);
    hX(k) =  plot(mean(mappedX(indizes,1)),mean(mappedX(indizes,2)),char(symbolsX(k)),'MarkerSize',12,'Color',cmap(k,:),'LineWidth',2); hold on;
    text(mean(mappedX(indizes,1)),mean(mappedX(indizes,2)),num2str(k));
end
legend(hX, numbersX);
hold off

embedding_2D_predictions = zeros(10,2);
for k = 1:10
    indizes = find(DaSet==k);
    embedding_2D_predictions(k,1:2) = mean(mappedX(indizes,1:2));
end

%% to be saved to a mat file
embedding_2D = mappedX(:,1:4);


%% embedding space of statistics of the timetraces
% from MetaParameterExtraction
load('Parameters174.mat');

Parameters174 = Parameters174(:,goodindizes);

ParametersXX = Parameters174;
for k = 1:10
    ixx = find(DaSet == k);
    ParametersXX(:,ixx) = repmat(mean(ParametersXX(:,ixx),2),[1 numel(ixx)]);
end
figure, imagesc(ParametersXX)
colormap(paruly)
caxis([-1.4 1.4])

ParametersXX = Parameters32;
for k = 1:10
    ixx = find(DaSet == k);
    ParametersXX(:,ixx) = repmat(mean(ParametersXX(:,ixx),2),[1 numel(ixx)]);
end
figure, imagesc(ParametersXX)
colormap(paruly)
caxis([-1.4 1.4])

[coeff,mappedY] = pca(Parameters174');

embedding_2D_stats = zeros(10,2);
Daset = DasetS(goodindizes);
for j = 1:10
    embedding_2D_stats(j,:) = median(mappedY(Daset==j,1:2));
end

symbolsX = {'o','x','s','d','^','p','h','*','+','>'};
numbersX = {'1','2','3','4','5','6','7','8','9','10'};
figure(312);
for k = 1:10
    plot(embedding_2D_stats(k,1),embedding_2D_stats(k,2),char(symbolsX(k)),'MarkerSize',12,'Color',cmap(k,:),'LineWidth',2); hold on;
    text(embedding_2D_stats(k,1),embedding_2D_stats(k,2),num2str(k));
end

load('Parameters32.mat');

% project into PCA space defined by the 174-neuron space
W = diag(std(Parameters32'))\coeff;
Projection33 = Parameters32'*W;

Daset = DasetS32;
embedding_2D_stats33 = zeros(5,2);
for j = 1:5
    embedding_2D_stats33(j,:) = median(Projection33(Daset==j,1:2));
end
figure(312);
for k = 1:5
    plot(embedding_2D_stats33(k,1),embedding_2D_stats33(k,2),char(symbolsX(k)),'MarkerSize',16,'Color',cmap(k,:),'LineWidth',2); hold on;
    text(embedding_2D_stats33(k,1),embedding_2D_stats33(k,2),num2str(k));
end



%% save all to a mat-file
save('embedding_spaces.mat','embedding_2D_predictions','embedding_2D_stats','embedding_2D_stats33','predicting_quality','goodindizes','DaSet2','DaSet');


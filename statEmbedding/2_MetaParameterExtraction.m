% Code written by Peter Rupprecht (2017), ptrrupprecht.wordpress.com
% The code calculates statistical parameters for each neuron and saves it
% to disk

clear Kurtouis Skewness Varianz Corr1sec Corr0sec Corr2sec HurstExp NoiseFreq DasetS
counter = 1;
for j = 1:10
    dataset = num2str(j);
    calcium_train = csvread([dataset '.train.calcium.csv']);
%     calcium_train = csvread([dataset '.test.calcium.csv']);
    
    for n = 1:size(calcium_train,2)
        L_trace = calcium_train(:,n);
        indizes = find(~isnan(L_trace) & ((L_trace~=0 | circshift(L_trace,1)~=0)) );
        L_trace = L_trace(indizes(2:end));
        
        Varianz(counter) = var(L_trace)/mean(L_trace);
        L_trace = (L_trace-median(L_trace))/std(L_trace);
        Kurtouis(counter) = kurtosis(L_trace);
        Skewness(counter) = skewness(L_trace);
        Corr0sec(counter) = corr(L_trace,circshift(L_trace,50));
        Corr1sec(counter) = corr(L_trace,circshift(L_trace,100));
        Corr2sec(counter) = corr(L_trace,circshift(L_trace,200));
        
        % hurst exponents
        HurstExp(counter,:) = genhurst(L_trace,[1:5]);
        
        % noise spectrum readout
        [PSD_noise,vect_freq_noise] = pwelch(L_trace-0*smooth(L_trace,40),1000,[],[],100);
        % cut off timetrace at the actual sampling frequency
        max_freq = find(abs(PSD_noise)<1e-4,1,'first');
        PSD_noise  = PSD_noise/sum(PSD_noise(1:max_freq-1));
        
        NoiseFreq(counter,:) = (-log(PSD_noise(1:6:37))-log(PSD_noise(2:6:38)))/2;
        
        DasetS(counter)  = j;
        counter = counter + 1;
    end
end

% Parameters174 is used for the training dataset (174 neurons)
% For the test dataset, the variable is renamed to Parameters33 (33
% neurons)

Parameters174 = [ (Varianz - mean(Varianz))/std(Varianz);
    (Kurtouis - mean(Kurtouis))/std(Kurtouis);
    (Skewness - mean(Skewness))/std(Skewness);
    (Corr0sec - mean(Corr0sec))/std(Corr0sec);
    (Corr1sec - mean(Corr1sec))/std(Corr1sec);
    (Corr2sec - mean(Corr2sec))/std(Corr2sec);
    (HurstExp(:,1)' - mean(HurstExp(:,1)))/std(HurstExp(:,1));
    (HurstExp(:,2)' - mean(HurstExp(:,2)))/std(HurstExp(:,2));
    (HurstExp(:,3)' - mean(HurstExp(:,3)))/std(HurstExp(:,3));
    (HurstExp(:,4)' - mean(HurstExp(:,4)))/std(HurstExp(:,4));
    (HurstExp(:,5)' - mean(HurstExp(:,5)))/std(HurstExp(:,5));
    (NoiseFreq(:,1)' - mean(NoiseFreq(:,1)))/std(NoiseFreq(:,1));
    (NoiseFreq(:,2)' - mean(NoiseFreq(:,2)))/std(NoiseFreq(:,2));
    (NoiseFreq(:,3)' - mean(NoiseFreq(:,3)))/std(NoiseFreq(:,3));
    (NoiseFreq(:,4)' - mean(NoiseFreq(:,4)))/std(NoiseFreq(:,4));
    (NoiseFreq(:,5)' - mean(NoiseFreq(:,5)))/std(NoiseFreq(:,5));
    (NoiseFreq(:,6)' - mean(NoiseFreq(:,6)))/std(NoiseFreq(:,6));
    (NoiseFreq(:,7)' - mean(NoiseFreq(:,7)))/std(NoiseFreq(:,7)) ];


save('Parameters174.mat','Parameters174','DasetS');


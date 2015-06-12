%% STEP 0: Initialize Parameters and Load Data

imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train
%training data
addpath ./common/;
images = loadMNISTImages('./common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('./common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%test data
testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%Initialize Parameters
Wc = 1e-1*randn(filterDim,filterDim,numFilters);
bc = zeros(numFilters, 1);

outDim = imageDim - filterDim + 1; % dimension of convolved image

outDim = outDim/poolDim;
hiddenSize = outDim^2*numFilters;

r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
bd = zeros(numClasses, 1);

%% STEP 1: Learn Parameters
epochs = 3;
alpha = 1e-1;
minibatch = 256;
% Setup for momentum
mom = 0.5;
momentum = .95;
momIncrease = 20;
Wc_velocity = zeros(size(Wc));
Wd_velocity = zeros(size(Wd));
bc_velocity = zeros(size(bc));
bd_velocity = zeros(size(bd));
lambda = 0.0001;
m = length(labels);

%%SGD loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = momentum;
        end;

        % get next randomly selected minibatch
        mb_data = images(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        numImages = length(mb_labels);
        convDim = imageDim-filterDim+1; % dimension of convolved output
        outputDim = (convDim)/poolDim; % dimension of subsampled output
        
        %Feedforward Propagation
        activations = cnnConvolve(filterDim, numFilters, mb_data, Wc, bc);%sigmoid(wx+b)
        activationsPooled = cnnPool(poolDim, activations);
        activationsPooled = reshape(activationsPooled,[],numImages);
        h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
        probs = bsxfun(@rdivide,h,sum(h,1));
        
        %Caculate Cost
        logp = log(probs);
        index = sub2ind(size(logp),mb_labels',1:size(probs,2));
        ceCost = -sum(logp(index));
        wCost = lambda/2 * (sum(Wd(:).^2)+sum(Wc(:).^2));
        cost = ceCost/numImages + wCost;
        
        %Backpropagation
        output = zeros(size(probs));
        output(index) = 1;
        DeltaSoftmax = probs - output;

        DeltaPool = reshape(Wd' * DeltaSoftmax,outputDim,outputDim,numFilters,numImages);
        DeltaUnpool = zeros(convDim,convDim,numFilters,numImages);

        for imNum = 1:numImages
            for FilterNum = 1:numFilters
                unpool = DeltaPool(:,:,FilterNum,imNum);
                DeltaUnpool(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim))./(poolDim ^ 2);
            end
        end

        DeltaConv = DeltaUnpool .* activations .* (1 - activations);
        
        Wd_grad = (1./numImages) .* DeltaSoftmax*activationsPooled'+lambda*Wd;
        bd_grad = (1./numImages) .* sum(DeltaSoftmax,2);

        bc_grad = zeros(size(bc));
        Wc_grad = zeros(filterDim,filterDim,numFilters);

        for filterNum = 1:numFilters
            error = DeltaConv(:,:,filterNum,:);
            bc_grad(filterNum) = (1./numImages) .* sum(error(:));
        end

        for filterNum = 1:numFilters
            for imNum = 1:numImages
                error = DeltaConv(:,:,filterNum,imNum);
                DeltaConv(:,:,filterNum,imNum) = rot90(error,2);
            end
        end

        for filterNum = 1:numFilters
            for imNum = 1:numImages
                Wc_grad(:,:,filterNum) = Wc_grad(:,:,filterNum) + conv2(mb_data(:,:,imNum),DeltaConv(:,:,filterNum,imNum),'valid');
            end
        end
        Wc_grad = (1./numImages) .* Wc_grad + lambda*Wc;
        %%% YOUR CODE HERE %%%
        Wc_velocity = mom*Wc_velocity+alpha*Wc_grad;
        Wc = Wc - Wc_velocity;
        Wd_velocity = mom*Wd_velocity+alpha*Wd_grad;
        Wd = Wd - Wd_velocity;
        bc_velocity = mom*bc_velocity+alpha*bc_grad;
        bc = bc - bc_velocity;
        bd_velocity = mom*bd_velocity+alpha*bd_grad;
        bd = bd - bd_velocity;
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;

    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;
    
    %test at each epoch
    activations = cnnConvolve(filterDim, numFilters, testImages, Wc, bc);%sigmoid(wx+b)
    activationsPooled = cnnPool(poolDim, activations);
    activationsPooled = reshape(activationsPooled,[],length(testLabels));
    h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
    probs = bsxfun(@rdivide,h,sum(h,1));
    [~,preds] = max(probs,[],1);
    preds = preds';
    acc = sum(preds==testLabels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
end;





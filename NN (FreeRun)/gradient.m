
function [Network, collmse] = gradient(L, n, m, smse, X, D, INEPOCH,GradientSearch, INI_W)

% 決定樣本數, 與輸出個數的維度
[P,N] = size(X);
[Pd,M] = size(D);

% input 樣本維度與 output 樣本維度不一致
if P ~= Pd 
    error('backprop:invalidTrainingAndDesired', ...
          'The number of input vectors and desired ouput do not match');
end

if length(L) < 3 % 防呆
    error('backprop:invalidNetworkStructure','The network must have at least 3 layers');
else
    if N ~= L(1) || M ~= L(end)
        e = sprintf('Dimensions of input (%d) does not match input layer (%d)',N,L(1));
        error('backprop:invalidLayerSize', e);
    elseif M ~= L(end)
        e = sprintf('Dimensions of output (%d) does not match output layer (%d)',M,L(end));
        error('backprop:invalidLayerSize', e);    
    end
end

% 權重初始化設定 %
nLayers = length(L); % we'll use the number of layers often  
w = cell(nLayers-1,1); % a weight matrix between each layer
rand('state',0);            % fix initial random variable
% 隨機權重的範圍 => -1 ~ +1.   

LayersIniW = INI_W+GradientSearch*(-INI_W/2);

for i=1:nLayers-2   
    w{i} = [LayersIniW - (LayersIniW .* 2 ) .* rand(L(i+1),L(i)+1) ;    zeros(1,L(i)+1)];
end
w{end} = LayersIniW - (LayersIniW .* 2) .* rand(L(end),L(end-1)+1);

% 訓練的終止條件設定
mse = Inf;  % 先給定mse為負無限大.
epochs = 0;

%%%%% PREALLOCATION PHASE %%%%%
a = cell(nLayers,1);  % one activation matrix for each layer
a{1} = [X ones(P,1)]; % a{1} 是包含input X 的資料, 後面再加一欄'1'的 bias

for i=2:nLayers-1
    a{i} = ones(P,L(i)+1);  
end
a{end} = ones(P,L(end));   % 在輸出層沒有 bias node


net = cell(nLayers-1,1); 
for i=1:nLayers-2;
    net{i} = ones(P,L(i+1)+1); % 固定 bias node 
end
net{end} = ones(P,L(end));

prev_dw = cell(nLayers-1,1);
sum_dw = cell(nLayers-1,1);
for i=1:nLayers-1
    prev_dw{i} = zeros(size(w{i})); % prev_dw starts at 0
    sum_dw{i} = zeros(size(w{i}));
end    

% training
while mse > smse && epochs < INEPOCH

    for i=1:nLayers-1
        net{i} = a{i} * w{i}'; % compute inputs to current layer

        if i < nLayers-1 % 隱藏層使用的轉移函數
            a{i+1} = [2./(1+exp(-1.*net{i}(:,1:end-1)))-1 ones(P,1)];
        else             % 輸出層使用的轉移函數
            a{i+1} =( 2 ./ (1 + exp(-0.5.*net{i}))) - 1;
        end
    end
    
    err = (D-a{end});       % 記錄每次訓練的預測誤差
    sse = sum(sum(err.^2)); % 
    
    % 由誤差 -> 計算(調整)輸出層的delta
    % S'(Output) * (D-Output) 其中, S'(Output) = (1+Output)*(1-Output)
    % 學習率 * 修正的誤差(delta) * 輸出值
    % S'(Activation) * ModifiedError * weight matrix,  其中, S'(Output) =
    % (1+Output)*(1-Output)
    delta = err .* (1 + a{end}) .* (1 - a{end});
    
    for i=nLayers-1:-1:1
        sum_dw{i} = n * delta' * a{i};
        if i > 1
            delta = (1+a{i}) .* (1-a{i}) .* (delta*w{i});
        end
    end
    
    % 更新 prev_w, weight matrices, epoch and mse
    for i=1:nLayers-1
        prev_dw{i} = (sum_dw{i} ./ P) + (m * prev_dw{i});
        w{i} = w{i} + prev_dw{i};
    end   
    epochs = epochs + 1;
    mse = sse/(P*M); % mse = 1/P * 1/M * summed squared error
    
    collmse(epochs,1) =  mse;
end

% return the trained network
Network.structure = L;
Network.weights = w;
Network.epochs = epochs;
Network.mse = mse;

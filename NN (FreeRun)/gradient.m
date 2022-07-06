
function [Network, collmse] = gradient(L, n, m, smse, X, D, INEPOCH,GradientSearch, INI_W)

% �M�w�˥���, �P��X�Ӽƪ�����
[P,N] = size(X);
[Pd,M] = size(D);

% input �˥����׻P output �˥����פ��@�P
if P ~= Pd 
    error('backprop:invalidTrainingAndDesired', ...
          'The number of input vectors and desired ouput do not match');
end

if length(L) < 3 % ���b
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

% �v����l�Ƴ]�w %
nLayers = length(L); % we'll use the number of layers often  
w = cell(nLayers-1,1); % a weight matrix between each layer
rand('state',0);            % fix initial random variable
% �H���v�����d�� => -1 ~ +1.   

LayersIniW = INI_W+GradientSearch*(-INI_W/2);

for i=1:nLayers-2   
    w{i} = [LayersIniW - (LayersIniW .* 2 ) .* rand(L(i+1),L(i)+1) ;    zeros(1,L(i)+1)];
end
w{end} = LayersIniW - (LayersIniW .* 2) .* rand(L(end),L(end-1)+1);

% �V�m���פ����]�w
mse = Inf;  % �����wmse���t�L���j.
epochs = 0;

%%%%% PREALLOCATION PHASE %%%%%
a = cell(nLayers,1);  % one activation matrix for each layer
a{1} = [X ones(P,1)]; % a{1} �O�]�tinput X �����, �᭱�A�[�@��'1'�� bias

for i=2:nLayers-1
    a{i} = ones(P,L(i)+1);  
end
a{end} = ones(P,L(end));   % �b��X�h�S�� bias node


net = cell(nLayers-1,1); 
for i=1:nLayers-2;
    net{i} = ones(P,L(i+1)+1); % �T�w bias node 
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

        if i < nLayers-1 % ���üh�ϥΪ��ಾ���
            a{i+1} = [2./(1+exp(-1.*net{i}(:,1:end-1)))-1 ones(P,1)];
        else             % ��X�h�ϥΪ��ಾ���
            a{i+1} =( 2 ./ (1 + exp(-0.5.*net{i}))) - 1;
        end
    end
    
    err = (D-a{end});       % �O���C���V�m���w���~�t
    sse = sum(sum(err.^2)); % 
    
    % �ѻ~�t -> �p��(�վ�)��X�h��delta
    % S'(Output) * (D-Output) �䤤, S'(Output) = (1+Output)*(1-Output)
    % �ǲ߲v * �ץ����~�t(delta) * ��X��
    % S'(Activation) * ModifiedError * weight matrix,  �䤤, S'(Output) =
    % (1+Output)*(1-Output)
    delta = err .* (1 + a{end}) .* (1 - a{end});
    
    for i=nLayers-1:-1:1
        sum_dw{i} = n * delta' * a{i};
        if i > 1
            delta = (1+a{i}) .* (1-a{i}) .* (delta*w{i});
        end
    end
    
    % ��s prev_w, weight matrices, epoch and mse
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

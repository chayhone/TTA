
function [OutPut] = gradient_Test(X,net)

w=net.weights;
L=net.structure;

[P,N] = size(X);

nLayers = length(L); % we'll use the number of layers often  

a = cell(nLayers,1);  
a{1} = [X ones(P,1)]; 
                      
for i=2:nLayers-1
    a{i} = ones(P,L(i)+1);
end
a{end} = ones(P,L(end));   

net = cell(nLayers-1,1); % one net matrix for each layer exclusive input

for i=1:nLayers-2;
    net{i} = ones(P,L(i+1)+1); % affix bias node 
end
net{end} = ones(P,L(end));


prev_dw = cell(nLayers-1,1);
sum_dw = cell(nLayers-1,1);
for i=1:nLayers-1
    prev_dw{i} = zeros(size(w{i})); % prev_dw starts at 0
    sum_dw{i} = zeros(size(w{i}));
end    

for i=1:nLayers-1
    net{i} = a{i} * w{i}'; % compute inputs to current layer

    if i < nLayers-1 % inner layers
        a{i+1} = [2./(1+exp(-1 .* net{i}(:,1:end-1)))-1 ones(P,1)];
    else             % output layers
        a{i+1} = (2 ./ (1 + exp(-0.5 .* net{i}))) - 1;
    end
end

OutPut=a{3,1};
    
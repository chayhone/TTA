function  [net, PSx, PSy, collmse] = BPNN_Train(create_X, create_y, nodes, epochs, MomTerm, Alpha,GradientSearch)
% up_threshold=15; %�W�W��
% low_threshold=9;%�U�W��
% coefficient = ( -log((up_threshold/low_threshold)-1) ) / low_threshold;%�w�q�D�X�ŦX��ƭp�⤧�`��

p1 = create_X(:,1:2:end);
t1 = create_y(:,1:2:end);

%�e�B�z�����ȻP�зǮt
[Z_X_value, PSx] = mapstd(p1);
%% �w��ISI���B�z
%         [a b]=find(Z_X_value>low_threshold);
%         Z_X_value(a,b)=up_threshold./ (1+exp(-coefficient*Z_X_value(a,b) ));
%         [c d]=find(Z_X_value<-low_threshold);
%         Z_X_value(c,d)=-(up_threshold./ (1+exp(-coefficient*abs(Z_X_value(c,d) ))));
%%
[Z_y_value0, PSy] = mapstd(t1);
Z_y_value = (2 ./ (1 + exp(-0.5.*Z_y_value0)) ) - 1;

[Z_y_i Z_y_j] = size(Z_y_value');
[Z_X_i Z_X_j] = size(Z_X_value');

INI_W = -0.05;

% ��פU���k (Gradient Descent)
[net, collmse] = gradient([Z_X_j nodes Z_y_j], Alpha, MomTerm, 0.01, Z_X_value', Z_y_value', epochs, GradientSearch, INI_W);




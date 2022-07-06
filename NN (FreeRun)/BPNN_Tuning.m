function [net, PSx, PSy, collmse] = BPNN_Tuning(create_X, create_y, net, nodes, epochs, MomTerm, Alpha,GradientSearch, spec_matrix)
% up_threshold=15; %�W�W��
% low_threshold=9;%�U�W��
% coefficient = ( -log((up_threshold/low_threshold)-1) ) / low_threshold;%�w�q�D�X�ŦX��ƭp�⤧�`��

%----------------------------�V�m�{��---------------------------
p1 = create_X(:, 1:end);
t1 = create_y(:, 1:end);

%�e�B�z�����ȻP�зǮt
[Z_X_value, PSx] = mapstd(p1);
%% �w��ISI���B�z
%         [a b]=find(Z_X_value>low_threshold);
%         Z_X_value(a,b)=up_threshold./ (1+exp(-coefficient*Z_X_value(a,b) ));
%         [c d]=find(Z_X_value<-low_threshold);
%         Z_X_value(c,d)=-(up_threshold./ (1+exp(-coefficient*abs(Z_X_value(c,d) ))));
% if isempty(spec_matrix)==0; 
%     [featuresize, samplesize]=size(Z_X_value); 
%    for i=1:featuresize
%                 up_threshold=spec_matrix(1,i); %�W�W��
%                 up_threshold_saturation  = up_threshold+6;  %�W���M�W��
%                 low_threshold=spec_matrix(2,i);%�U�W��
%                 low_threshold_saturation=spec_matrix(2,i)-6;% �U���M�W��
% %                 coefficient = ( -log((up_threshold/low_threshold)-1) ) / low_threshold;%�w�q�D�X�ŦX��ƭp�⤧�`��
%                [a,b]=find(Z_X_value(i,:)>up_threshold);
%                   Z_X_value(a,i)=up_threshold+ ((1./ (1+power(1.1,-(Z_X_value(a,i)-up_threshold)))    -1/2)*2)*abs(up_threshold_saturation-up_threshold);%�H1.4����
% %                         Z_X_value(a,i)=up_threshold+ ((1./ (1+exp(-(Z_X_value(a,i)-up_threshold)))    -1/2)*2)*abs(up_threshold_saturation-up_threshold);%�He����
%                 %                     Z_X_Running(a,b)=up_threshold./ (1+exp(-coefficient*Z_X_Running(a,b) ));
%                [c,d]=find(Z_X_value(i,:)<low_threshold);
% %                         Z_X_value(c,i)=low_threshold+ ((1./ (1+exp(- (Z_X_value(c,i)- low_threshold) ))-1/2)*2)*abs(low_threshold_saturation-low_threshold); %�He����
%                  Z_X_value(c,i)=low_threshold+ ((1./ (1+power(1.1,- (Z_X_value(c,i)- low_threshold) ))-1/2)*2)*abs(low_threshold_saturation-low_threshold);
%     end 
% end
%%
[Z_y_value1, PSy] = mapstd(t1);
Z_y_value = 2 ./ (1 + exp(-0.5 .* Z_y_value1)) - 1;

[Z_y_i Z_y_j] = size(Z_y_value');
[Z_X_i Z_X_j] = size(Z_X_value');

INI_W = -0.05;
[net, collmse] = gradient_tune([Z_X_j nodes Z_y_j], net, Alpha, MomTerm, 0.01, Z_X_value', Z_y_value', epochs,GradientSearch, INI_W);

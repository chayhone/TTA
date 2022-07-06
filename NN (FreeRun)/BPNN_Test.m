function BPNN_Predictive = BPNN_Test(net, PSx, create_X, PSy, create_y)
%----------------------------���յ{��---------------------------
%����Jp�P�ؼ�t
p1p = create_X(:,1:end); 

%�e�B�z�s����J-->�����ȻP�зǮt
pn1p = mapstd('apply', p1p, PSx);

an1p1 = gradient_Test(pn1p', net);
an1p = (-log( (2./(an1p1+1)) - 1) ) ./ (1 .* 0.5);


BPNN_temp = mapstd('reverse', an1p', PSy)';
%%���e��
% if ( BPNN_temp<0.0005 ) %�o��n�p�� ��0.0015����n�_
%     BPNN_temp = 0.001;
% end
% % for others
% if(BPNN_temp > 0.02)
%      BPNN_temp = 0.02;
% end

%%���e�� 20121022����
% if ( BPNN_temp<0.5 ) %�o��n�p�� ��0.0015����n�_
%     BPNN_temp = 1;
% end
% for others
% if(BPNN_temp > 22)
%      BPNN_temp = 20;
% end



BPNN_Predictive = BPNN_temp;

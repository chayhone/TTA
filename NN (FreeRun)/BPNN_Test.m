function BPNN_Predictive = BPNN_Test(net, PSx, create_X, PSy, create_y)
%----------------------------測試程式---------------------------
%取輸入p與目標t
p1p = create_X(:,1:end); 

%前處理新的輸入-->平均值與標準差
pn1p = mapstd('apply', p1p, PSx);

an1p1 = gradient_Test(pn1p', net);
an1p = (-log( (2./(an1p1+1)) - 1) ) ./ (1 .* 0.5);


BPNN_temp = mapstd('reverse', an1p', PSy)';
%%門檻值
% if ( BPNN_temp<0.0005 ) %這邊要小心 用0.0015比較好否
%     BPNN_temp = 0.001;
% end
% % for others
% if(BPNN_temp > 0.02)
%      BPNN_temp = 0.02;
% end

%%門檻值 20121022關掉
% if ( BPNN_temp<0.5 ) %這邊要小心 用0.0015比較好否
%     BPNN_temp = 1;
% end
% for others
% if(BPNN_temp > 22)
%      BPNN_temp = 20;
% end



BPNN_Predictive = BPNN_temp;

function [OutBPNN_Phase_I] = MCS_BPNN_FreeRun(InNet, inPSx, inPSy, InRunX)

% tic % �p�� MR Running�ɶ����_�I

% TotalnewX = InRunX;     
% TotalnewY = InRuny;   
% Target = InTarget;  

PSx = inPSx;
net = InNet;
PSy = inPSy;
% ModelNum = InModelNum;

% �ססססססססססססס׸�ƫe�B�z �ססססססססססססס�
% temp = ['load ' InModelRoute 'BPNN_Model_Stage2_' int2str(ModelNum) '.mat '];  evalc(temp);
% [MCSFreeRunKVI AVMFreeRuneKVI] = LoadFreeRunKVI(InModelRoute,ModelNum);
% FilterKVIStd=MCSFreeRunKVI(:,Choose_Sensor_ID);
% [TotalnewX_i TotalnewX_j] = size(TotalnewX); % TotalnewY_i=����

%----------------------------���յ{��---------------------------
%�e�B�z�s����J-->�����ȻP�зǮt
% newX = TotalnewX(:, FilterKVIStd);

newX = InRunX;
pn1p = mapstd('apply', newX', PSx);

an1p1 = gradient_Test(pn1p', net);

an1p = (-log( (2./(an1p1+1)) - 1) ) ./ (1 .* 0.5);

OutBPNN_Phase_I = mapstd('reverse', an1p', PSy)';


% ====================================================================
% �I�I�I�I�I �p��MAPE �I�I�I�I�I
% for NowLot  = 1: size(TotalnewY,1)
%     OutBPNN_PhaseI_Error(NowLot,:)  = (abs(OutBPNN_Phase_I(NowLot,:) - TotalnewY(NowLot,:)));%./ Target(1,:)) .* 100;
%     mape_temp(NowLot,:) = OutBPNN_PhaseI_Error(NowLot,:) ;
% end

% OutBPNN_PhaseI_MaxErr = max(OutBPNN_PhaseI_Error);
% OutBPNN_PhaseI_MAPE   = mean(mape_temp)  ;
% ====================================================================

% OutBPNN_FreeRunSpend = toc; % �p��MR�ؼҮɶ������I
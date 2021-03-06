function [OutBPNN_Phase_I] = MCS_BPNN_FreeRun(InNet, inPSx, inPSy, InRunX)

% tic % 計算 MR Running時間之起點

% TotalnewX = InRunX;     
% TotalnewY = InRuny;   
% Target = InTarget;  

PSx = inPSx;
net = InNet;
PSy = inPSy;
% ModelNum = InModelNum;

% ＝＝＝＝＝＝＝＝＝＝＝＝＝＝資料前處理 ＝＝＝＝＝＝＝＝＝＝＝＝＝＝
% temp = ['load ' InModelRoute 'BPNN_Model_Stage2_' int2str(ModelNum) '.mat '];  evalc(temp);
% [MCSFreeRunKVI AVMFreeRuneKVI] = LoadFreeRunKVI(InModelRoute,ModelNum);
% FilterKVIStd=MCSFreeRunKVI(:,Choose_Sensor_ID);
% [TotalnewX_i TotalnewX_j] = size(TotalnewX); % TotalnewY_i=筆數

%----------------------------測試程式---------------------------
%前處理新的輸入-->平均值與標準差
% newX = TotalnewX(:, FilterKVIStd);

newX = InRunX;
pn1p = mapstd('apply', newX', PSx);

an1p1 = gradient_Test(pn1p', net);

an1p = (-log( (2./(an1p1+1)) - 1) ) ./ (1 .* 0.5);

OutBPNN_Phase_I = mapstd('reverse', an1p', PSy)';


% ====================================================================
% ＠＠＠＠＠ 計算MAPE ＠＠＠＠＠
% for NowLot  = 1: size(TotalnewY,1)
%     OutBPNN_PhaseI_Error(NowLot,:)  = (abs(OutBPNN_Phase_I(NowLot,:) - TotalnewY(NowLot,:)));%./ Target(1,:)) .* 100;
%     mape_temp(NowLot,:) = OutBPNN_PhaseI_Error(NowLot,:) ;
% end

% OutBPNN_PhaseI_MaxErr = max(OutBPNN_PhaseI_Error);
% OutBPNN_PhaseI_MAPE   = mean(mape_temp)  ;
% ====================================================================

% OutBPNN_FreeRunSpend = toc; % 計算MR建模時間之終點

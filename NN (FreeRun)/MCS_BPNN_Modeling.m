% function [OutBPNN_Stage_I, OutStageI_BPNN_Error, OutStageI_BPNN_MaxError,OutStageI_BPNN_MAPE,...
%     OutBPNN_Stage_II,OutStageII_BPNN_Error,OutStageII_BPNN_MaxError,OutStageII_BPNN_MAPE,...
%     OutBPNN_RT,OutBPNN_CreateModelSpend]...
%     = MCS_BPNN_Modeling(IncellKeepIndex,IncellKeyVariableIndex,InX,Iny,InTarget,InModelNum,InTempRoute,InModelRoute)
function [net, PSx, PSy] = MCS_BPNN_Modeling(InX, InY)

% tic % get start time
% cellKeepIndex = IncellKeepIndex;  
X = InX';    
y = InY';    
% Target = InTarget;     
% ModelNum = InModelNum;
% cellKeyVariableIndex = IncellKeyVariableIndex;
% % ＝＝＝ KSS ＝＝＝
% ArrKeepSiteIndex = find(cellKeepIndex{1,1} == 1); % Is's to find lot Keep Site Index
% X = X(ArrKeepSiteIndex,:);
% y = y(ArrKeepSiteIndex,:);
% % ＝＝＝ KVI ＝＝＝
% ArrKeyVariableIndex = cellKeyVariableIndex{1,1};
% KVI_X = X(:, ArrKeyVariableIndex);

create_y = y;
create_X = X;
% [create_X, Choose_Sensor_ID] = ClearStdZeroSensor(KVI_X); % delete X value std = 0
% FilterKVIStd = ArrKeyVariableIndex(Choose_Sensor_ID);
% iniStage2_ConjectureSTD = 999999999;

% load BPNN_Parameters.mat (The Main_SetParameters.m Create this file, by yaya)
%  temp = ['load ' InModelRoute 'BPNN_Parameters.mat ']; evalc(temp);

% ------------ 複製在 nntool的設定 ------------
% epochs = 10000
% goal = 0.0025
% min_grad = 0.000001
% max_fail = 10000
% Ir = 0.01
% Ir_dec = 0.7
% Ir_inc = 1.05
% max_perf_inc = 1.04

epochsRange = 10000;
mcLowLim = 0.1;
mcTic = 0.5;
mcUpLim = 1;
AlphaLowLim =0.15;
AlphaTic = 0.1;
AlphaUpLim = 0.15;
nodesRange= ceil(size(X, 1)/2);

for epochs = epochsRange % (1) epoch
    iniMAPE = 10000;
    for MomTerm = mcLowLim : mcTic: mcUpLim % (2)  MomTerm
        for Alpha = AlphaLowLim : AlphaTic : AlphaUpLim % 0.15 : 0.1 : 0.15  % (3)  Alpha
            for nodes = nodesRange %(nodesRange(1,1)-5) : 1 : nodesRange(1,2)  % (4) Nodes
                GradientSearch = 0;%%if GradientSearch =0 ==>使用初始設定的初始權重；GradientSearch =1==>若以初始權重求算後的預測值為Inf則開啟收詢初始權重之機制
                TuningSwich = 1;%1:進行Tunning，0：搜尋完畢不在進行Tunning
                while(TuningSwich == 1)
                    [net, PSx, PSy, collmse] = BPNN_Train(create_X, create_y, nodes, epochs, MomTerm, Alpha,GradientSearch);
                    BPNN_Predictive = BPNN_Test(net, PSx, create_X, PSy);
                    if(any(isinf(BPNN_Predictive))==1) %NN送出Inf值
                        GradientSearch = GradientSearch+1; %開啟收詢初始權重之機制
                        TuningSwich = 1; %1:繼續進行Tunning
                    else
                        TuningSwich = 0;%0：搜尋完畢不再進行Tunning
                    end
                end % end of while(TuningSwich == 1)
%                 [OutStageI_BPNN_Error, OutStageI_BPNN_MaxError, OutStageI_BPNN_MAPE, ...
%                     OutBPNN_Stage_I, OutBPNN_RT] = calculate(BPNN_Predictive, y, Target);
%                 sortOutStageI_BPNN_MAPE = sort(OutStageI_BPNN_MAPE, 'descend');
%                 if mean(sortOutStageI_BPNN_MAPE(1,1+fix(0.5*size(create_y,1)))) <= iniMAPE
%                     iniMAPE = mean(OutStageI_BPNN_MAPE);
%                     temp = ['save ' InModelRoute 'BPNN_tempModel_Stage1_' int2str(ModelNum) '.mat '];
%                     temp = [temp 'X  y	net Choose_Sensor_ID create_X create_y Target '];
%                     temp = [temp 'net PSx PSy nodes epochs MomTerm Alpha '];
%                     temp = [temp 'OutStageI_BPNN_Error OutStageI_BPNN_MaxError '];
%                     temp = [temp 'OutStageI_BPNN_MAPE OutBPNN_Stage_I OutBPNN_RT '];
%                     evalc(temp);
%                 end
            end
        end
    end
%     temp = ['load ' InModelRoute 'BPNN_tempModel_Stage1_' int2str(ModelNum) '.mat ']; evalc(temp);

    GradientSearch = 0;%%if GradientSearch =0 ==>使用初始設定的初始權重；GradientSearch =1==>若以初始權重求算後的預測值為Inf則開啟收詢初始權重之機制
    TuningSwich = 1;%1:進行Tunning，0：搜尋完畢不在進行Tunning
    while(TuningSwich == 1)  %STAGE II HERE
        [net, PSx, PSy] = BPNN_Tuning(create_X, create_y, net, nodes, epochs, MomTerm, Alpha,GradientSearch,[]);  %記得一定要丟個空矩陣進去
        BPNN_Predictive = BPNN_Test(net, PSx, create_X, PSy);
        if(any(isinf(BPNN_Predictive))==1) %NN送出Inf值
            GradientSearch = GradientSearch+1; %開啟收詢初始權重之機制
            TuningSwich = 1; %1:繼續進行 Tunning
        else
            TuningSwich = 0;%0：搜尋完畢不再進行 Tunning
        end
    end % end of while(TuningSwich == 1)
%     %-----------------------------------------------------------------------------------------------------%
%     % 以防StageI計算出的OutBPNN_RT被StageII計算出的OutBPNN_RT覆蓋
%     %，故將OutBPNN_RT改為OutBPNN_RT_II   20101213 by Water
%     %-----------------------------------------------------------------------------------------------------%
%     [OutStageII_BPNN_Error, OutStageII_BPNN_MaxError, OutStageII_BPNN_MAPE,...
%         OutBPNN_Stage_II, OutBPNN_RT_II] = calculate(BPNN_Predictive, y, Target);
% 
%     Stage2_ConjectureSTD  = std(BPNN_Predictive*100);
% 
%     temp = ['save ' InModelRoute 'BPNN_tempModel_Stage2_' int2str(ModelNum) '.mat '];
%     temp = [temp 'X  y	net Choose_Sensor_ID create_X create_y Target '];
%     temp = [temp 'OutStageII_BPNN_Error OutStageII_BPNN_MaxError  '];
%     temp = [temp 'OutStageII_BPNN_MAPE OutBPNN_Stage_II OutBPNN_RT BPNNRefreshCounter'];
%     evalc(temp);
% 
%     sortStage2_ConjectureSTD = sort(Stage2_ConjectureSTD, 'descend');
%     if mean(sortStage2_ConjectureSTD(1,1+fix(0.5*size(create_y,1)))) <= iniStage2_ConjectureSTD
%         iniStage2_ConjectureSTD = mean(sortStage2_ConjectureSTD(1,1+fix(0.5*size(create_y,1))));
%         temp = ['save ' InModelRoute 'iniStage2_ConjectureSTD iniStage2_ConjectureSTD'];  evalc(temp);
% 
%         temp = ['load ' InModelRoute 'BPNN_tempModel_Stage1_' int2str(ModelNum) '.mat '];  evalc(temp);
% 
% 
%         temp = ['save ' InModelRoute 'BPNN_Model_Stage1_' int2str(ModelNum) '.mat '];
%         temp = [temp 'X  y	net Choose_Sensor_ID create_X create_y Target '];
%         temp = [temp 'net PSx PSy nodes epochs MomTerm Alpha '];
%         temp = [temp 'OutStageI_BPNN_Error OutStageI_BPNN_MaxError '];
%         temp = [temp 'OutStageI_BPNN_MAPE OutBPNN_Stage_I OutBPNN_RT '];
%         evalc(temp);
% 
%         temp = ['load ' InModelRoute 'BPNN_tempModel_Stage2_' int2str(ModelNum) '.mat ']; evalc(temp);
% 
%         RefreshTimes=0; IsRefreshOK_Index=0;  Refresh_Error = [];
%         historyRefreshContinuous = 0;
%         temp = ['save ' InModelRoute 'BPNN_Model_Stage2_' int2str(ModelNum) '.mat '];
%         temp = [temp 'X  y	net Choose_Sensor_ID create_X create_y Target '];
%         temp = [temp 'net PSx PSy nodes epochs MomTerm Alpha '];
%         temp = [temp 'RefreshTimes IsRefreshOK_Index Refresh_Error '];
%         temp = [temp 'OutStageII_BPNN_Error OutStageII_BPNN_MaxError  '];
%         temp = [temp 'OutStageII_BPNN_MAPE OutBPNN_Stage_II OutBPNN_RT historyRefreshContinuous BPNNRefreshCounter'];
%         evalc(temp);
% 
%         temp = ['load ' InModelRoute 'iniStage2_ConjectureSTD'];  evalc(temp);
%     end
end
% 
% temp = ['load ' InModelRoute 'BPNN_Model_Stage1_' int2str(ModelNum) '.mat '];  evalc(temp);
% temp = ['load ' InModelRoute 'BPNN_Model_Stage2_' int2str(ModelNum) '.mat '];  evalc(temp);
% 
% temp = ['delete ' InModelRoute 'BPNN_tempModel_Stage1_' int2str(ModelNum) '.mat '];  evalc(temp);
% temp = ['delete ' InModelRoute 'BPNN_tempModel_Stage2_' int2str(ModelNum) '.mat '];  evalc(temp);
% temp = ['delete ' InModelRoute 'iniStage2_ConjectureSTD.mat'];  evalc(temp);
% 
% OutBPNN_CreateModelSpend = toc; % 計算MR建模時間之終點





% ======= source code backup ==========
% if size(create_y, 2) > 100 % BigSize (大筆數)
%    [mcLowLim mcTic mcUpLim epochsRange nodeRange] = BPNN_Setting('BigSize');
% else                              % SmallSize (小筆數)
%    [mcLowLim mcTic mcUpLim epochsRange nodeRange] = BPNN_Setting('SmallSize');
% end

%nodeLowBound = fix((size(Choose_Sensor_ID,2)+size(create_y,1))./2) - nodeRange;
% nodeUpBound  = fix((size(Choose_Sensor_ID,2)+size(create_y,1))./2) + nodeRange;
%nodeLowBound = fix((size(Choose_Sensor_ID,2))./2) - nodeRange;
%nodeUpBound  = fix((size(Choose_Sensor_ID,2))./2) + nodeRange;

function [BPNNtestError, BPNNtestMaxError, BPNN_MAPE,...
    BPNN_Predictive, OutBPNN_RT] = calculate(BPNN_Predictive, y, Target)

% ====================================================================
for NowLot  = 1: size(y,1)
    BPNNtestError(NowLot,:) = (abs(BPNN_Predictive(NowLot,:) - y(NowLot,:)));%./ Target(1,:)) .* 100;
    BPNNError = BPNNtestError  ;
end
BPNNtestMaxError = max(BPNNtestError,[],1);
BPNN_MAPE = mean(BPNNError,1);

% ====================================================================

%-----------------------------------------------------------------------------------------------------%
%(1)BPNNtestError������Model�ӼơA���]StageI�O�H���Ƥ��ؼҰ�Ƥ����աA
%    �G�]�H���Ƶ����(BPNNtestError_even)�p��Refresh Threshold
%(2)����Refresh Threshold�p��覡�A��Max Error-Min Error�אּMax Error+Min Error
%20101213 by Water 
%-----------------------------------------------------------------------------------------------------%
BPNNtestError_even = BPNNtestError(2:2:end,:);
OutBPNN_RT = (max(BPNNtestError_even) + min(BPNNtestError_even) ) ./ 2;
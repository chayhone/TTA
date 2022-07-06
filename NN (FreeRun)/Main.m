
% 清理執行環境
clc;
clear;

% 設定建模樣本數量 (測試樣本的數量將從全樣本數量扣除建模樣本數量)
Modeling_Size = 40;
% 把重要參數貼進來
%Feature_Index = [1 2 3 4 5 6 7 8 9 10 11 12 13];
Feature_Index = [1 2 3 4];
% 1 2  6 7 8 9 10 11 12 13

% =========== 以下就不用管了，預測結果在變數 "ReturnVal (左欄為預測值，右欄為實際值)" ===========
% 資料載入
load('CollectX.txt');  % 所有樣本的參數特徵值，內容值不須先標準化
load('CollectY.txt');  % 所有輪框樣本的偏擺量測之標準差值，不須先標準化

% 這裡會根據在 NSGA-II所挑選的重要參數之編號 (Feature_Index)撈出X內的參數，所以不須另外先挑哦 !!!
SaveFinger = 1;
X =[];
[temp, L] = size(Feature_Index);
for i = 1 : size(CollectX, 2)
    for j = 1 : L
        if i == Feature_Index(1, j)
            X(:, SaveFinger) = CollectX(:, i);
            SaveFinger  = SaveFinger + 1;
        end
    end
end
Y = CollectY;

% 將陣列 X與陣列 Y拆解為建模樣本與測試樣本
Modeling_X = X(1 : Modeling_Size, :);
Modeling_Y = Y(1 : Modeling_Size, :);
Running_X = X((Modeling_Size + 1) : size(X, 1), :);  % 剩下的當作測試樣本
Running_Y = Y((Modeling_Size + 1) : size(X, 1), :);  % 剩下的當作測試樣本

% 建立類神經的預測模型
[net, PSx, PSy] = MCS_BPNN_Modeling(Modeling_X, Modeling_Y);  

% 使用新建立的 NN模型計算測試樣本
ReturnVal = [];
for i = 1 : size(Running_X, 1)
	% 逐一取出測試樣本
    newX = Running_X(i, :);  % 該樣本的參數特徵值未經標準化
    ReturnVal(i, 1) = MCS_BPNN_FreeRun(net, PSx, PSy, newX);    % 預測值
    ReturnVal(i, 2) = Running_Y(i, :);   %原來的實際值
    ReturnVal(i, 3) = abs((ReturnVal(i, 1)-ReturnVal(i, 2))/ReturnVal(i, 2)); % 絕對誤差百分比
end

Count_C = 0;
Count_F = 0;
for  i = 1 : size(Running_X, 1)
    if ReturnVal(i, 1) > 0.33 && ReturnVal(i, 2) > 0.33
        Count_C = Count_C + 1;
    end
    if ReturnVal(i, 1) < 0.33 && ReturnVal(i, 2) < 0.33
        Count_C = Count_C + 1;
    end    
    if ReturnVal(i, 1) > 0.33 && ReturnVal(i, 2) < 0.33
        Count_F = Count_F + 1;
    end
    if ReturnVal(i, 1) < 0.33 && ReturnVal(i, 2) > 0.33
        Count_F = Count_F + 1;
    end            
end


% ErrorSum = ['MAPE(%) of Modeling: ' num2str(mean(ReturnVal(1:Modeling_Size, 3))*100) '; MAPE(%) of Testing: ' num2str(mean(ReturnVal(Modeling_Size:size(ReturnVal, 1), 3))*100)]


% 結果在陣列 "ReturnVal"，記得要存。    ^ ^"





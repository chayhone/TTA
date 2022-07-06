
% �M�z��������
clc;
clear;

% �]�w�ؼҼ˥��ƶq (���ռ˥����ƶq�N�q���˥��ƶq�����ؼҼ˥��ƶq)
Modeling_Size = 40;
% �⭫�n�ѼƶK�i��
%Feature_Index = [1 2 3 4 5 6 7 8 9 10 11 12 13];
Feature_Index = [1 2 3 4];
% 1 2  6 7 8 9 10 11 12 13

% =========== �H�U�N���κޤF�A�w�����G�b�ܼ� "ReturnVal (���欰�w���ȡA�k�欰��ڭ�)" ===========
% ��Ƹ��J
load('CollectX.txt');  % �Ҧ��˥����ѼƯS�x�ȡA���e�Ȥ������зǤ�
load('CollectY.txt');  % �Ҧ����ؼ˥������\�q�����зǮt�ȡA�������зǤ�

% �o�̷|�ھڦb NSGA-II�ҬD�諸���n�ѼƤ��s�� (Feature_Index)���XX�����ѼơA�ҥH�����t�~���D�@ !!!
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

% �N�}�C X�P�}�C Y��Ѭ��ؼҼ˥��P���ռ˥�
Modeling_X = X(1 : Modeling_Size, :);
Modeling_Y = Y(1 : Modeling_Size, :);
Running_X = X((Modeling_Size + 1) : size(X, 1), :);  % �ѤU����@���ռ˥�
Running_Y = Y((Modeling_Size + 1) : size(X, 1), :);  % �ѤU����@���ռ˥�

% �إ������g���w���ҫ�
[net, PSx, PSy] = MCS_BPNN_Modeling(Modeling_X, Modeling_Y);  

% �ϥηs�إߪ� NN�ҫ��p����ռ˥�
ReturnVal = [];
for i = 1 : size(Running_X, 1)
	% �v�@���X���ռ˥�
    newX = Running_X(i, :);  % �Ӽ˥����ѼƯS�x�ȥ��g�зǤ�
    ReturnVal(i, 1) = MCS_BPNN_FreeRun(net, PSx, PSy, newX);    % �w����
    ReturnVal(i, 2) = Running_Y(i, :);   %��Ӫ���ڭ�
    ReturnVal(i, 3) = abs((ReturnVal(i, 1)-ReturnVal(i, 2))/ReturnVal(i, 2)); % ����~�t�ʤ���
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


% ���G�b�}�C "ReturnVal"�A�O�o�n�s�C    ^ ^"





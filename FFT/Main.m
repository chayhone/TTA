
% 清理執行環境
clear;
clc;
% 載入數據
RawData = load('D:\2015_04_02\201504021837_第三次加工\Adam4716Dev2\(1000)103.txt');
SampleRate = 10240;
RPM = 5000;
null_column = 8;

H = size(RawData, 1);
StartPoint = floor(H / 2) - (10240 * 1);
EndPoint = floor(H / 2) + (10240 * 0);
CalculateData = RawData(StartPoint : EndPoint, 13 + null_column);
% 計算參數的特徵值

Feature_Indicator = CalculateFeatureIndicator(CalculateData, SampleRate, RPM);
%Feature_Indicator2 = CalculateFeatureIndicator2(ggg, 20480, 2000);




% �M�z��������
clear;
clc;
% ���J�ƾ�
RawData = load('D:\2015_04_02\201504021837_�ĤT���[�u\Adam4716Dev2\(1000)103.txt');
SampleRate = 10240;
RPM = 5000;
null_column = 8;

H = size(RawData, 1);
StartPoint = floor(H / 2) - (10240 * 1);
EndPoint = floor(H / 2) + (10240 * 0);
CalculateData = RawData(StartPoint : EndPoint, 13 + null_column);
% �p��Ѽƪ��S�x��

Feature_Indicator = CalculateFeatureIndicator(CalculateData, SampleRate, RPM);
%Feature_Indicator2 = CalculateFeatureIndicator2(ggg, 20480, 2000);



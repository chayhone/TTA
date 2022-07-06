clear;
clc;
% 載入資料
START =1;
c = 1001;
SampleRate = 2000;
RawData = load('all.mat');
RawData_FineMachining = RawData.all(START:START+SIZE-1,1);
RawData_NoMachining = RawData.all(START:START+SIZE-1,2);

% 取得資料長度
[C,R] = size(RawData_NoMachining);
% 若採樣頻率為每秒2000筆，那麼每筆資料僅占 1/2000秒
TimeLength = [];
for i = 1 : SIZE
    TimeLength(i, 1) = i/(SampleRate);
end
% 取出 C區間的時間
X = [];
X = TimeLength(1:SIZE, 1);
% 僅取 Channel 0
RawData_FineMachining_Ch0 = RawData_FineMachining(:, 1);
RawData_NoMachining_Ch0 = RawData_NoMachining(:, 1);

% 未濾波版
RMS_FineMachining_Ch0 = RMS(RawData_FineMachining_Ch0);  %加工未濾
RMS_NoMachining_Ch0 = RMS(RawData_NoMachining_Ch0);  %未加工未濾

% 有濾波版
DenoisedRMS_FineMachining_Ch0 = RMS(Denoise(RawData_FineMachining_Ch0)); %加工有濾
DenoisedRMS_NoMachining_Ch0 = RMS(Denoise(RawData_NoMachining_Ch0)); %未加工有濾

SNR(1) = 20 * log(power(RMS_FineMachining_Ch0, 2) / power(RMS_NoMachining_Ch0, 2));  %加工未濾/未加工未濾
SNR(2) = 20 * log(power(DenoisedRMS_FineMachining_Ch0, 2) / power(DenoisedRMS_NoMachining_Ch0, 2)); %加工濾波/未加工濾波
% Draw
figure(1);
subplot(2,1,1);
hold on;
xlabel('Time (sec)');
ylabel('G (m/s2)');
title('No Machining')
%axis([0,0.1,-0.1,0.1]);
plot(X(:, 1),RawData_NoMachining_Ch0, '-b');
plot(X(:, 1),Denoise(RawData_NoMachining_Ch0), '-r');
legend('Raw Signal','Denoised Signal');
text(0.1,0.3,['SNR: ',num2str(SNR(1)),' (dB)']);

%figure(2);
subplot(2,1,2);
hold on;
xlabel('Time (sec)');
ylabel('G (m/s2)');
title('Fine Machining')
%axis([0,0.1,-0.1,0.1]);
plot(X(:, 1),RawData_FineMachining_Ch0, '-b');
plot(X(:, 1),Denoise(RawData_FineMachining_Ch0), '-r');
legend('Raw Signal','Denoised Signal');
text(0.1,1.8,['SNR: ',num2str(SNR(2)),' (dB)']);

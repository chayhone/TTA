clear;
clc;
% ���J���
START =1;
c = 1001;
SampleRate = 2000;
RawData = load('all.mat');
RawData_FineMachining = RawData.all(START:START+SIZE-1,1);
RawData_NoMachining = RawData.all(START:START+SIZE-1,2);

% ���o��ƪ���
[C,R] = size(RawData_NoMachining);
% �Y�ļ��W�v���C��2000���A����C����ƶȥe 1/2000��
TimeLength = [];
for i = 1 : SIZE
    TimeLength(i, 1) = i/(SampleRate);
end
% ���X C�϶����ɶ�
X = [];
X = TimeLength(1:SIZE, 1);
% �Ȩ� Channel 0
RawData_FineMachining_Ch0 = RawData_FineMachining(:, 1);
RawData_NoMachining_Ch0 = RawData_NoMachining(:, 1);

% ���o�i��
RMS_FineMachining_Ch0 = RMS(RawData_FineMachining_Ch0);  %�[�u���o
RMS_NoMachining_Ch0 = RMS(RawData_NoMachining_Ch0);  %���[�u���o

% ���o�i��
DenoisedRMS_FineMachining_Ch0 = RMS(Denoise(RawData_FineMachining_Ch0)); %�[�u���o
DenoisedRMS_NoMachining_Ch0 = RMS(Denoise(RawData_NoMachining_Ch0)); %���[�u���o

SNR(1) = 20 * log(power(RMS_FineMachining_Ch0, 2) / power(RMS_NoMachining_Ch0, 2));  %�[�u���o/���[�u���o
SNR(2) = 20 * log(power(DenoisedRMS_FineMachining_Ch0, 2) / power(DenoisedRMS_NoMachining_Ch0, 2)); %�[�u�o�i/���[�u�o�i
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

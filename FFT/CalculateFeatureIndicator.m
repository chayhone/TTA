function result = CalculateFeatureIndicator(x, inSampleRate, inRPM)
    % ��X�γ~
    indicator = [];   
    % �ϥΤp�i�o�����T -------------------
    lev=5;
%     x = x';    % ���`�p��ɡA�ݱN������ѡC�����sĶ�� DLL�A�h���������ѡC    
    denoise = wden(x,'sqtwolog', 's', 'mln' , lev, 'db3'); 
    % �W�v�� -------------------------------
    % ���o��ƪ���
    [M, L] = size(denoise);
    % �ֳt�ť߸��ഫ
    y = fft(denoise, M);
    % �ର�b�U�W�v�W���񪺯�q
    p = abs(y) / M;
    % �p��U�W�v�b�}�C�W��������m
    F= inSampleRate * linspace(0, 1, M);
    scale = F(1, 2);
    % �W�q�e��
    BoundWidth = 5;  
    % ���W
    F_base = inRPM / 60;
    % �p��b 1/4 ���W�����񪺯�q
	LowLimit = round(((F_base * 0.25) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % �w���W�X�}�C�d��
    end    
	TopLimit = round(((F_base * 0.25) + BoundWidth) / scale);
	indicator(1, 1) = sum(p(LowLimit : TopLimit));
	% �p��b 1/2 ���W�����񪺯�q
	LowLimit = round(((F_base * 0.5) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % �w���W�X�}�C�d��
    end    
	TopLimit = round(((F_base * 0.5) + BoundWidth) / scale);
	indicator(2, 1) = sum(p(LowLimit : TopLimit));
	% �p��b 1 ���W�����񪺯�q
	LowLimit = round(((F_base * 1) - BoundWidth) / scale);      
    if LowLimit <= 0
        LowLimit = 1;    % �w���W�X�}�C�d��
    end        
	TopLimit = round(((F_base * 1) + BoundWidth) / scale);
	indicator(3, 1) = sum(p(LowLimit : TopLimit));        
	% �p��b 2 ���W�����񪺯�q
	LowLimit = round(((F_base * 2) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % �w���W�X�}�C�d��
    end             
	TopLimit = round(((F_base * 2) + BoundWidth) / scale);
	indicator(4, 1) = sum(p(LowLimit : TopLimit));
	% �p��b 3 ���W�����񪺯�q
	LowLimit = round(((F_base * 3) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % �w���W�X�}�C�d��
    end             
	TopLimit = round(((F_base * 3) + 1) / scale);
	indicator(5, 1) = sum(p(LowLimit : TopLimit));           
    % �ɶ��� -------------------------------
    % �зǮt
	indicator(6, 1) = std(denoise);
    % ���A�Y��
    indicator(7, 1) = skewness(denoise);
    % �p�A�Y��
    indicator(8, 1) = kurtosis(denoise);
    %indicator(8, 1) = 0;
    % ����ڴ� (RMS)
    RMS_Value = sqrt(sum(denoise.*conj(denoise))/size(denoise,1));      
    indicator(10, 1) = RMS_Value;
    % CF
    indicator(9, 1) = (max(denoise)./(RMS_Value));  
    % ������
    indicator(11, 1) = mean(denoise);
    % ���j��    
    indicator(12, 1) = max(denoise);
    % ���p��
    indicator(13, 1) = min(denoise);
    % �p�����
    indicator(14, 1) = max(denoise)-min(denoise);
    % �p�i�] (�����p�i)
    wpt = wpdec(denoise, 5, 'db5','shannon');
    wp_energy_percentage = wenergy(wpt);
    indicator(15, 1) = wp_energy_percentage(1, 1);
    indicator(16, 1) = wp_energy_percentage(1, 2);
    indicator(17, 1) = wp_energy_percentage(1, 3);
    indicator(18, 1) = wp_energy_percentage(1, 4);
    indicator(19, 1) = wp_energy_percentage(1, 5);
    indicator(20, 1) = wp_energy_percentage(1, 6);
    indicator(21, 1) = wp_energy_percentage(1, 7);
    indicator(22, 1) = wp_energy_percentage(1, 8);
    indicator(23, 1) = wp_energy_percentage(1, 9);  
    indicator(24, 1) = wp_energy_percentage(1, 10);     
    indicator(25, 1) = wp_energy_percentage(1, 11);    
    indicator(26, 1) = wp_energy_percentage(1, 12);    
    indicator(27, 1) = wp_energy_percentage(1, 13);    
    indicator(28, 1) = wp_energy_percentage(1, 14);    
    indicator(29, 1) = wp_energy_percentage(1, 15);    
    indicator(30, 1) = wp_energy_percentage(1, 16);    
    % �N�Ѽƪ��S�x�ȿ�X
    result = indicator;    
end

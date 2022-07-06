function result = CalculateFeatureIndicator(x, inSampleRate, inRPM)
    % 輸出用途
    indicator = [];   
    % 使用小波濾除雜訊 -------------------
    lev=5;
%     x = x';    % 平常計算時，需將此行註解。但為編譯成 DLL，則須取消註解。    
    denoise = wden(x,'sqtwolog', 's', 'mln' , lev, 'db3'); 
    % 頻率域 -------------------------------
    % 取得資料長度
    [M, L] = size(denoise);
    % 快速傅立葉轉換
    y = fft(denoise, M);
    % 轉為在各頻率上釋放的能量
    p = abs(y) / M;
    % 計算各頻率在陣列上的對應位置
    F= inSampleRate * linspace(0, 1, M);
    scale = F(1, 2);
    % 頻段寬放
    BoundWidth = 5;  
    % 基頻
    F_base = inRPM / 60;
    % 計算在 1/4 倍頻所釋放的能量
	LowLimit = round(((F_base * 0.25) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % 預防超出陣列範圍
    end    
	TopLimit = round(((F_base * 0.25) + BoundWidth) / scale);
	indicator(1, 1) = sum(p(LowLimit : TopLimit));
	% 計算在 1/2 倍頻所釋放的能量
	LowLimit = round(((F_base * 0.5) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % 預防超出陣列範圍
    end    
	TopLimit = round(((F_base * 0.5) + BoundWidth) / scale);
	indicator(2, 1) = sum(p(LowLimit : TopLimit));
	% 計算在 1 倍頻所釋放的能量
	LowLimit = round(((F_base * 1) - BoundWidth) / scale);      
    if LowLimit <= 0
        LowLimit = 1;    % 預防超出陣列範圍
    end        
	TopLimit = round(((F_base * 1) + BoundWidth) / scale);
	indicator(3, 1) = sum(p(LowLimit : TopLimit));        
	% 計算在 2 倍頻所釋放的能量
	LowLimit = round(((F_base * 2) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % 預防超出陣列範圍
    end             
	TopLimit = round(((F_base * 2) + BoundWidth) / scale);
	indicator(4, 1) = sum(p(LowLimit : TopLimit));
	% 計算在 3 倍頻所釋放的能量
	LowLimit = round(((F_base * 3) - BoundWidth) / scale);
    if LowLimit <= 0
        LowLimit = 1;    % 預防超出陣列範圍
    end             
	TopLimit = round(((F_base * 3) + 1) / scale);
	indicator(5, 1) = sum(p(LowLimit : TopLimit));           
    % 時間域 -------------------------------
    % 標準差
	indicator(6, 1) = std(denoise);
    % 偏態係數
    indicator(7, 1) = skewness(denoise);
    % 峰態係數
    indicator(8, 1) = kurtosis(denoise);
    %indicator(8, 1) = 0;
    % 均方根植 (RMS)
    RMS_Value = sqrt(sum(denoise.*conj(denoise))/size(denoise,1));      
    indicator(10, 1) = RMS_Value;
    % CF
    indicator(9, 1) = (max(denoise)./(RMS_Value));  
    % 平均值
    indicator(11, 1) = mean(denoise);
    % 極大值    
    indicator(12, 1) = max(denoise);
    % 極小值
    indicator(13, 1) = min(denoise);
    % 峰對蜂值
    indicator(14, 1) = max(denoise)-min(denoise);
    % 小波包 (離散小波)
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
    % 將參數的特徵值輸出
    result = indicator;    
end

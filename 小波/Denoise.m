function[outValue] = Denoise(inRawData)
    lev = 5;       
    outValue = wden(inRawData,'sqtwolog', 's', 'mln' , lev, 'db3');     % 使用小波濾除雜訊
end
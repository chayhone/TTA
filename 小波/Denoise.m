function[outValue] = Denoise(inRawData)
    lev = 5;       
    outValue = wden(inRawData,'sqtwolog', 's', 'mln' , lev, 'db3');     % �ϥΤp�i�o�����T
end
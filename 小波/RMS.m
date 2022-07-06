function RMS_arr = RMS(Data)
RMS_arr = sqrt(sum(Data.*conj(Data))/size(Data,1));

CF_arr = (max(Data)./(RMS_arr));
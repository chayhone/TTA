function[outFeatureIndex] = deleteFeature(inFeatureIndex, inModelingX, inModelingY)
    Temp = [];
	MAPE_Del = [];
	for j = 1 : size(FeatureIndex, 2)
        Temp = FeatureIndex(1, j);
        FeatureIndex(1, j) = 9999;
        MAPE_Del(j, 1) = getMapeByNN(FeatureIndex, inModelingX, inModelingY);
        FeatureIndex(1, j) = Temp;
    end
	for j = 1 : size(MAPE_Del, 1)
        if MAPE_Del(j, 1) <= MAPE_Old
            FeatureIndex(1, j) = 9999;
        end
    end
	Temp = [];
	SaveFinger = 1;
	for i = 1 : size(FeatureIndex, 2)
        if FeatureIndex(1, i) ~= 9999
            Temp(1, SaveFinger) = FeatureIndex(1, i);
            SaveFinger = SaveFinger + 1;
        end
    end
	outFeatureIndex = Temp;    
end
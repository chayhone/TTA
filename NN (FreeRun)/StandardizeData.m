function [Z_X_value,PSx,Z_y_value,PSy,spec_matrix] = StandardizeData(create_X,create_y)

% 將全數點的X做標準化
X = [create_X(:,1:1:end)];
[Z_X_value, PSx] = mapstd(X);  % [Z_X_value1,X_mean,X_std] = prestd(X);

 [feature_size, temp]=size(Z_X_value);
for i=1:feature_size
    max_ISI = max(Z_X_value(i,:) );
    min_ISI = min(Z_X_value(i,:));
    max_ISI_int=ceil(max_ISI);
    min_ISI_int=floor(min_ISI);
    spec_matrix(1,i)=   max_ISI_int;
    spec_matrix(2,i)=   min_ISI_int;
end
Z_X_value = Z_X_value';

% 將y做標準化
[Z_y_value, PSy] = mapstd(create_y); % [Z_y_value,y_mean,y_std] = prestd(create_y);
Z_y_value = Z_y_value';





 
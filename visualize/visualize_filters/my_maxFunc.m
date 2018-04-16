function [optX] = my_maxFunc(f, x0, options)

optX = x0;
%f_scat(optX);

for i = 1:options.maxIter
  disp(['iter_num ' num2str(i)])  
  gd = num_grad(f, optX, 1e-4);
  optX = (optX+gd)/norm(optX+gd);  
  %f_scat(optX)
end
end


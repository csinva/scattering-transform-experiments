function [ y ] = f_scat( v, k )
% get one output of scat
filt_opt = struct;
scat_opt.M = 1;
Wop = wavelet_factory_2d([8 8],filt_opt, scat_opt);
x = reshape(v, 8,8);

[S, ~] = scat(x, Wop);

y = S{2}.signal{k};

end


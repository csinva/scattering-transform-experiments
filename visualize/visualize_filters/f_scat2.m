function [ y ] = f_scat2( v, k )
% get one output of scat
filt_opt = struct;
scat_opt.M = 2;
Wop = wavelet_factory_2d([8 8],filt_opt, scat_opt);
x = reshape(v, 8,8);

[S, ~] = scat(x, Wop);

y = S{3}.signal{k};

end


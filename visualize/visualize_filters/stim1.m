
% max the stimilus that maximize S{3}.signal{k}
cd('../../');
addpath_scatnet;
cd('test/yuansi/');

load('MNIST.mat');

k = 5;
N = 60000;

mat = zeros(64, N);
Sfs = zeros(417, N);
[Wop,filters] = wavelet_factory_2d([28,28]);
parfor i = 1:N,
    x = reshape(X(:,i), 28, 28);
    a = randi(21);b = randi(21);
    x = x(a:a+7, b:b+7);
	x = x - mean(x(:));
	x = x/norm(x, 'fro');
    mat(:, i) = reshape(x, 64, 1);
    [S,U] = scat(x,Wop);
    Sf = format_scat(S);
    Sfs(:, i) = Sf;
    
end

save('stim_mn_60000.mat','mat','Sfs', '-v7.3');

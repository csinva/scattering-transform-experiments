
% max the stimilus that maximize S{3}.signal{k}
cd('../../');
addpath_scatnet;
cd('test/yuansi/');

load('MNIST.mat');

k = 5;
N = 600000;
batch = 60000;

mat = zeros(64, N);
Sfs = zeros(417, N);
[Wop,filters] = wavelet_factory_2d([28,28]);
matJ = zeros(64, batch);
SfsJ = zeros(64, batch);
for j = 1:10,
parfor i = 1:batch,
    x = reshape(X(:,i), 28, 28);
	a = randi(21);b = randi(21);
    x = x(a:a+7, b:b+7);
	x = x/norm(x, 'fro');
    matJ(:, i) = reshape(x, 64, 1);
    [S,U] = scat(x,Wop);
    Sf = format_scat(S);
    SfsJ(:, i) = Sf;
end
	mat(:, (j-1)*batch+1:j*batch) = matJ;
    Sfs(:, (j-1)*batch+1:j*batch) = SfsJ;	
end

save('stim_norm_600000.mat','mat','Sfs', '-v7.3');

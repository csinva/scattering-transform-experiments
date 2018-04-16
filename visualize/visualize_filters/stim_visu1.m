cd('../../');
addpath_scatnet;
cd('test/yuansi/');

load('stim_visu/v_l2_f1.mat');
addpath('numerical');

H = num_hess(@(x) f_scat(x), v, 1000, 1e-3);
save('stim_visu/H_l2_f1.mat','H');

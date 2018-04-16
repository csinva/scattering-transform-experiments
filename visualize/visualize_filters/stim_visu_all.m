%% addpaths
addpath 'scatnet';
addpath_scatnet;
addpath 'numerical'
addpath('numerical');

%% run
options.maxIter = 20;
scores = zeros(32, 10);

for i = 1:1 % was 32
	for j = 1:1 % 2
      disp([i j])
      
      % calculate gradient
      X = rand(64, 1);
      optX = my_maxFunc(@(x) f_scat(x, i), X, options);
	  scores(i,j) = f_scat(optX, i);
	  parsave(sprintf('out/v_l2_f%d_t%d.mat', i, j), optX);
      
      % calculate Hessian
	  % H = num_hess(@(x) f_scat(x,i), optX, 1e-3);
	  % parsave(sprintf('out/H_l2_f%d_t%d.mat', i, j), H);
      
    end
end

disp('saving...')
save('out/scores.mat', 'scores');
disp('finished!')

%% visualize
x = load(sprintf('out/v_l2_f%d_t%d.mat', 1, 1)); % loads optX
x = x.x;
z = reshape(x, [8 8]);
imshow(z)


function H = forward_hess(func, X, h)
H = zeros(length(X));

% for each dimension of objective function
for i=1:length(X)
x1 = X;
x1(i) = X(i) + 1.5*h;

df1 = num_grad(func, x1, 0.5*h);

x2 = X;
x2(i) = X(i) + 0.5*h;
df2 = num_grad(func, x2, 0.5*h);

d2f = (df1-df2) / (h);

% assign as row i of Hessian
H(i,:) = d2f';
end
end


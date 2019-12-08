% input filepath
% output accuracy
% example usage: FOGD('C:\Users\qi\Desktop\new_final\code\QiCheng\data\poplar\poplar')
function output = FOGD(path)
example_file = strcat(path,'.features.txt');
Y = double(load(example_file));
row = size(Y,1);
X = Y(:,2:end);
Y = Y(:,end);
w_v = find(Y(:) == 0);
for w_i = 1:length(w_v)
    Y(w_v(w_i)) = -1;
end
B=50;
sigma  =64;
eta_fou=0.002;
D=4;
eta    = eta_fou;
ID = generateID(length(Y), 20);
err_count = 0;
B = B*D;
w = zeros(1, B*2);
u = (1/sigma)*randn(size(X,2),B);
for t = 1:length(ID)
    id = ID(t);
    x_t = X(id,:);
    nx_t = [cos(u'*x_t'); sin(u'*x_t')];
    
    f_t=w*nx_t;
    err = sign(f_t);
    if (err==0)
        err=1;
    end

    y_t=Y(id);
    if (err~=y_t)
        err_count = err_count + 1;
    end
    if Y(id)*f_t < 1
        w = w + eta*y_t*nx_t';
    end
end
output = 1-(err_count/row)
end

function [ID] = generateID(n, t)
ID=[];
for i=1:t
    ID = [ID; randperm(n)];
end
end
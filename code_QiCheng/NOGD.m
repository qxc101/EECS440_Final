function output = NOGD(path)
example_file = strcat(path,'.features.txt');
true_label = double(load(example_file));
row = size(true_label,1);
features = true_label(:,2:end);
true_label = true_label(:,end);
w_v = find(true_label(:) == 0);
for w_i = 1:length(w_v)
    true_label(w_v(w_i)) = -1;
end
params.Budget = 10;
params.eta = 0.2;
params.sigma  =64;
params.eta_nogd = 0.2;
params.k = 0.2;
eta = params.eta_nogd;
B = params.Budget;
k = params.k * B;
ID = generateID(length(true_label), 20);
err_count = 0;
alpha = [];
vector = [];
flag = 0;
tic
for t = 1:length(ID)
    id = ID(t);
    if (isempty(alpha))
        f_t=0;
    else
        k_t = kernal(features, params, id, vector);
        if(flag == 0)
        f_t=alpha*k_t;
        else
        nx_t = M*k_t;
        f_t=w*nx_t;
        end
    end
    hat_y_t = sign(f_t);
    if (hat_y_t==0)
        hat_y_t=1;
    end
    y_t=true_label(id);
    if (hat_y_t~=y_t)
        err_count = err_count + 1;
    end
    if true_label(id)*f_t < 1
        if (size(alpha,2)<B)
            alpha = [alpha eta*y_t];
            vector = [vector id];
        else
            if (flag == 0)
            k_hat = kernal(features, params, vector, vector);
            [V,D] = eigs(k_hat, k);
            M=D^(-0.5)*V';
            flag = 1;
            w = alpha*pinv(M);
            nx_t = M*k_t;
            w = w + eta*y_t*nx_t';
            else

            w = w + eta*y_t*nx_t';
            end
        end
    end
end
output = 1-(err_count/row);
end

function [ID] = generateID(n, t)
ID=[];
for i=1:t
    ID = [ID; randperm(n)];
end
end

function [k] = kernal(features, params, id, SV)
   gid = sum(features(id,:).*features(id,:),2);
   gsv = sum(features(SV,:).*features(SV,:),2);
   gidsv = features(id,:)*features(SV,:)';
   k = exp(-(repmat(gid',length(SV),1) + repmat(gsv,1,length(id)) - 2*gidsv')/(2*params.sigma^2));
end

function namw = RY(path,p)
alabel_file = strcat(path,'.response.txt');
tlabel_file = strcat(path,'.gold.txt');
example_file = strcat(path,'.features.txt');
insID_file = strcat(path,'.ins_ids.txt');
%example_file
worker_response = double(load(alabel_file));
true_label = double(load(tlabel_file));
n_i=size(true_label,1)*p;
for a = 1:n_i
    temp_r = randi([1,384]);
    if true_label(temp_r,2) == 1
        true_label(temp_r,2) = 0;
    else
        true_label(temp_r,2) = 1;
    end
end
features = double(load(example_file));
features = features(:,1:(end-1));
instance_ids = double(load(insID_file));
%pi = fw(w,xi)
namw.o1 = worker_response;
namw.o2 = true_label;
worker_ids =unique(worker_response(:,1));
alpha_array = zeros(size(worker_ids));
beta_array = zeros(size(worker_ids));
res_array = cell(size(worker_ids));
for a = 1:size(worker_ids)
    [alpha,beta,res_i] = alpha_beta(worker_ids(a),worker_response,true_label);
    alpha_array(a) = alpha;
    beta_array(a) = beta;
    res_array{a} = res_i;
end

namw.o3 = alpha_array;
namw.o4 = beta_array;
namw.o5 = features;
namw.o6 = instance_ids;
w = zeros(1,64);
for m = 1:64
    w(m) = 1;
end
y = features(1,:);
u = U(0,alpha_array,beta_array,res_array,w,y);
namw.o7 = MV(1,worker_response);
u_array = zeros(1,length(instance_ids));
for y = 1:length(u_array)
    u_array(y) = MV(instance_ids(y),worker_response);
end

namw.o6 = MV(0,worker_response);
degree = 1;
target =  true_label(:,2);
data = features;
for a = 1:n_i
    temp_r = randi([1,384]);
    if true_label(temp_r,2) == 1
        true_label(temp_r,2) = 0;
    else
        true_label(temp_r,2) = 1;
    end
end
target(target > 1) = 0;
prev_error=0;
rows = size(data,1);
cols = size(data,2);
Hmatrix = zeros(rows,1);

for row = 1: rows
    Hmatrix(row, 1) = 1;
    y = 2;
    for col = 1: cols
        for deg = 1:degree
            Hmatrix(row, y) = u_array(col)*data(row, col);
            y = y+1;
        end
    end
end

condition = true;
wght = zeros(cols*degree+1,1);
phi_trans = transpose(Hmatrix);
m = 1;

while condition
    wghtT = transpose(wght);
    namw = zeros(rows,1);
    for i = 1:rows
        namw(i,1) = wghtT * phi_trans(1:end,i);
        namw(i,1) = 1 / (1 + exp(namw(i,1)*(-1)));
    end
    E = phi_trans * (namw - target);
    new_error = sum(E,1);
    error_diff = abs(new_error - prev_error);
    R = zeros(rows,rows);
    for i = 1:rows
        R(i,i) = namw(i,1) * (1 - namw(i,1));
    end
    new_wght = wght - pinv(phi_trans * R * Hmatrix) * E ;
    
    condition = abs(sum(new_wght) - sum(wght)) >= 0.001 && error_diff>= 0.001;
    if condition
        wght = new_wght;
        prev_error = new_error;
    end
    m = m + 1;
    temp_ab = update_alphabeta(u_array,worker_ids,res_array,instance_ids);
    alpha_array = temp_ab.alpha;
    beta_array = temp_ab.beta;
    
    for u_index = 1:size(u_array,1)
        temp_data = [features(u_index,:) true_label(u_index)];
        u_array(u_index) = U(i,alpha_array,beta_array,res_array,wght,temp_data);
    end
end
target = true_label;
test_data = features;
rows = size(test_data,1);
cols = size(test_data,2);
Hmatrix = zeros(rows,1);
for row = 1: rows
    Hmatrix(row, 1) = 1;
    y = 2;
    for col = 1: cols
        for deg = 1:degree
            Hmatrix(row, y) = test_data(row, col)^deg;
            y = y+1;
        end
    end
end
phi_trans = transpose(Hmatrix);
for i = 1:rows
    namw(i,1) = transpose(new_wght) * phi_trans(1:end,i);
    namw(i,1) = 1 / (1 + exp(namw(i,1)*(-1)));
end

target(target > 1) = 0;
predicted = zeros(size(namw, 1), 1);
accuracy = zeros(rows, 1);

for i = 1:rows
    first = transpose(new_wght) * transpose(Hmatrix(i, 1:end));
    second = namw(i, 1) ;
    if (first > 0) && (second > 0.5)
        predicted(i, 1) = 1;
        if predicted(i, 1)  == target(i, 1)
            accuracy(i, 1) = 1;
        end
    elseif (first < 0) && (1 - second > 0.5)
        predicted(i, 1) = 0;
        namw(i, 1) = (1 - second);
        if predicted(i, 1) == target(i, 1)
            accuracy(i, 1) = 1;
        end
    else
        predicted(i, 1) = 1;
        accuracy(i, 1) = 0.5;
    end
    
end

num = sum(accuracy);
den = size(accuracy, 1);
final_acc = num/den;

fprintf('classification accuracy=%6.4f \n', final_acc)
namw = final_acc;


end



function output = update_alphabeta(all_ui,worker_ids,res_array,instance_ids)
a1 = 1;
a2 = 1;
b1 = 1; 
b2  = 1;
for j = 1:size(worker_ids,1)
    
    temp = res_array{j};
    sum_a1 = 0;
    sum_a2 = 0;
    sum_b1 = 0;
    sum_b2 = 0;
    for i = 1;size(instance_ids,1)
        temp_index = find(temp(:,1) == instance_ids(i));
        if size(temp_index,1) == 0
            sum_a1 = sum_a1 + all_ui(i)*0;
            sum_b1 = sum_b1 + (1-all_ui(i))*(0);
        else
            sum_a1 = sum_a1 + all_ui(i)*temp(temp_index);
            sum_b1 = sum_b1 + (1-all_ui(i))*(1-temp(temp_index));
        end
        sum_a2 = sum_a2 + all_ui(i);
        sum_b2 = sum_b2 + (1 - all_ui(i));
    end
    alpha(j) = (a1 - 1 + sum_a1)/(a2+a1-2+sum_a2);
    beta(j) = (b1 - 1 + sum_b1)/(b2+b1-2+sum_b2);
end
output.alpha = alpha;
output.beta = beta;
end


function output = MV(j,worker_response)
R = find(worker_response(:,2) == j);
R = size(R,1);
sum = 0;
ids = find(worker_response(:,2) == j);
for j = 1:length(ids)
    sum = sum + worker_response(ids(j),3);
end
output = (1/R)*sum;
end

function output = E(alpha_array,beta_array,res_array,w,features,instance_ids)
e = 0;

leng = size(features);
for i = 1:leng(1)
    id = instance_ids(i);
    u = U(id,alpha_array,beta_array,res_array,w,features(:,i));
    [p,a,b] = pab(i,alpha_array,beta_array,res_array,w,features(:,i));
    e = e + (u*log(p*a) + (1-u)*log*(1-p)*b);
end
output = e;
end


function output = U(i,alpha_array,beta_array,res_array,w,x)
[p,a,b] = pab(i,alpha_array,beta_array,res_array,w,x);
output = a*p/(a*p+b*(1-p));
end

function [p,a,b] = pab(i,alpha_array,beta_array,res_array,w,x)
o_a=1;
o_b = 1;
p = fw(w,x);
for index = 1:length(beta_array)
    
    temp = res_array{index};
    yi = find(temp(:,1) == i);
    if  size(yi,1) == 0
        o_a = o_a*(1);
        o_b = o_b*(1);
    else
        o_a = o_a*(1);
        o_b = o_b*(1);
        temp_a = alpha_array(index);
        temp_b = beta_array(index);
        o_a = o_a*((temp_a^yi)*(temp_a^(1-yi)));
        o_b = o_b*((temp_b^(1-yi))*(temp_b^yi));
    end
end

a = o_a;
b = o_b;

end

%give weight and attribute vector
function output = fw(w,x)
wt = transpose(w);
wtx = dot(wt,x);
output = 1/(1+exp(wtx));
end

%output speci or sensi

function [alpha,beta,res_i] = alpha_beta(j,a_label,t_label)

tp_count = sum(t_label(:,2) == 1);
tn_count = sum(t_label(:,2) == 0);
len = size(a_label);
ano_count = sum(a_label(:,1) == j);
response = a_label(ano_count);
response_index = 1;
for worker_i = 1:len(1)
    if a_label(worker_i,1) == j
        response(response_index,1) = a_label(worker_i,2);
        response(response_index,2) = a_label(worker_i,3);
        response_index = response_index + 1;
    end
end
len = size(response);
pp_count = 0;
nn_count = 0;
for instance_i=1:len(1)
    tlabel_index = find(t_label(:,1) == response(instance_i,1));
    if response(instance_i,2) == t_label(tlabel_index,2) && response(instance_i,2) == 1
        pp_count = pp_count + 1;
    end
    if response(instance_i,2) == t_label(tlabel_index,2) && response(instance_i,2) == 0
        nn_count = nn_count + 1;
    end
end

alpha = pp_count/tp_count;
beta = nn_count/tn_count;
res_i = response;
end

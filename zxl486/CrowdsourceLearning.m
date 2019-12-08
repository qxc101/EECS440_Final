

path='D:\CourseWork\EECS440FinalProject\data\alder\alder';
[featStack,trueLabels,insID,workerID,~,noisyLabels]=dataParser(path);
[holdout_feats,holdout_trueLabels,...
    data,data_trueLabels,data_ID,...
    data_noisyLabels,data_mvLabels]=data_setup(featStack,trueLabels,insID,noisyLabels);

type_sizes=zeros(length(data(1,:)),1);
data=kBins(3,data,type_sizes);

CSwithAL(data,data_mvLabels);


function CSwithAL(data,labels)

type_sizes=ones(length(data(1,:)),1);

n=length(labels);
L_idx=randsample(n,round(n*0.3));
L_data=data(L_idx,:);
L_labels=labels(L_idx,1);
U_idx=transpose(setdiff(1:n,L_idx));
U_data=data(U_idx,:);
U_labels=labels(U_idx,1);

learner1=bayes(L_data,L_labels,type_sizes,3,-1,-1);

results=testAll(data,learner1.model);
Acc1=sum(results1(:,1)==labels)/length(labels);



end

function query_simple



end


function results=testAll(test_data,model)
results=zeros(size(test_data,1),4);
for x=1:size(test_data,1)
    [logpos,logneg]=test(test_data(x,:),model);
    pos=exp(logpos);
    neg=exp(logneg);
    Pneg=neg/(pos+neg);
    Ppos=pos/(pos+neg);
    results(x,3)=Ppos;
    results(x,4)=Pneg;
    if Ppos>Pneg
        results(x,1)=1;
        results(x,2)=Ppos;
    else
        results(x,1)=0;
        results(x,2)=Pneg;
    end
end
end


function [holdout_feats,holdout_trueLabels,...
    data,data_trueLabels,data_ID,data_noisyLabels,...
    data_mvLabels]=data_setup(featStack,trueLabels,insID,noisyLabels)
n=length(featStack(:,1));
holdout_idx=randsample(n,round(n*0.3));
holdout_feats=featStack(holdout_idx,:);
holdout_trueLabels=trueLabels(holdout_idx);
data_idx=transpose(setdiff(1:n,holdout_idx));
data=featStack(data_idx,:);
data_trueLabels=trueLabels(data_idx);
data_ID=insID(data_idx);
data_noisyLabels=[];
for i=1:length(data_ID)
    data_noisyLabels=[data_noisyLabels;noisyLabels(noisyLabels(:,2)==data_ID(i),:)];
end
% Majority vote
data_mvLabels=zeros(length(data_ID),2);
for j=1:length(data_ID)
    insLabels=data_noisyLabels(data_noisyLabels(:,2)==data_ID(j),3);
    pcount=sum(insLabels==1);
    ncount=sum(insLabels==0);
    if pcount>ncount
        data_mvLabels(j,1)=1;
        data_mvLabels(j,2)=pcount/(pcount+ncount);
    elseif pcount==ncount
        data_mvLabels(j,1)=randi([0 1]);
        data_mvLabels(j,2)=0.5;
    else
        data_mvLabels(j,1)=0;
        data_mvLabels(j,2)=ncount/(pcount+ncount);
    end
end
end

function [featStack,trueLabels,insID,workerID,workerLabels,noisyLabels]=dataParser(path)
load([path,'.features.txt']);
load([path,'.gold.txt']);
load([path,'.ins_ids.txt']);
load([path,'.response.txt']);
featStack=alder_features(:,1:end-1);
trueLabels1=alder_features(:,end);
trueLabels2=alder_gold(:,end);
if trueLabels1==trueLabels2
    trueLabels=trueLabels1;
    clear trueLabels1 trueLabels2
else
    error('True labels cannot be determined! Check dataset.');
end
insID=alder_ins_ids;
workerID=unique(alder_response(:,1));
workerLabels=cell([1 length(workerID)]);
for i=1:length(workerID)
    workerLabels{i}=alder_response(alder_response(:,1)==workerID(i),:);
end
noisyLabels=alder_response;
end




%% Subfunction for Naive Bayes
% Subfunction: constructing a bayes probability model
function output = bayes(ATTRIBUTES,CLASSIFICATIONS,TYPE_SIZES,k,m,weight)
colnum=max(TYPE_SIZES);
if sum(TYPE_SIZES==0)>0
    ATTRIBUTES=kBins(k,ATTRIBUTES,TYPE_SIZES);
    if k>max(TYPE_SIZES)
        colnum=k;
    end
end
n_model = zeros(length(TYPE_SIZES)+1, colnum*2);
p_count = sum(CLASSIFICATIONS(:,1) == 1);
n_count = sum(CLASSIFICATIONS(:,1) == 0);
py_p = p_count/length(CLASSIFICATIONS);
py_n = n_count/length(CLASSIFICATIONS);
n_model(1,1) = py_p;
n_model(1,2) = py_n;
temp_p = ATTRIBUTES;
temp_n = ATTRIBUTES;
temp_p_weight=weight;
temp_n_weight=weight;
temp_label = CLASSIFICATIONS;
i1 = 1;
while i1 <= length(temp_label)
    if temp_label(i1) == 1
        temp_label(i1) = [];
        temp_n(i1,:) = [];
        if length(temp_n_weight)>1
            temp_n_weight(i1)=[];
        end
    else
        i1 = i1+1;
    end
end
temp_label = CLASSIFICATIONS;
i2 = 1;
while i2 <= length(temp_label)
    if temp_label(i2) == 0
        temp_label(i2) = [];
        temp_p(i2,:) = [];
        if length(temp_p_weight)>1
            temp_p_weight(i2)=[];
        end
    else
        i2 = i2+1;
    end
end
i3 = 1;
while i3 <= (length(TYPE_SIZES))
    v=length(unique(ATTRIBUTES(:,i3)));
    p=1/v;
    if m < 0
        m = v;
    end
    for i4 = 1:v
        if length(weight)==1 && weight<0
            n_model(i3+1,i4) = (sum(temp_p(:,i3) == i4)+m*p)/(p_count+m);
        else
            choice1=temp_p(:,i3) == i4;
            n_model(i3+1,i4) = (sum(choice1.*temp_p_weight)+m*p)/(sum(temp_p_weight)+m);
        end
        
    end
    i3 = i3+1;
end
i5 = 1;
while i5 <= (length(TYPE_SIZES))
    v=length(unique(ATTRIBUTES(:,i5)));
    p=1/v;
    if m < 0
        m = v;
    end
    for i6 = 1:v
        if length(weight)==1 && weight<0
            n_model(i5+1,colnum+i6) = (sum(temp_n(:,i5) == i6)+m*p)/(n_count+m);
        else
            choice2=temp_n(:,i5) == i6;
            n_model(i5+1,colnum+i6) = (sum(choice2.*temp_n_weight)+m*p)/(sum(temp_n_weight)+m);
        end
    end
    i5 = i5+1;
end
output.model = n_model;
output.discAttr = ATTRIBUTES;
end

function discreteAttributes=kBins(k,attributes,type_sizes)
% Subfunction: kBins discretizing continuous attributes
% Discretize continuous attributes. Partition the range of feature into k
% bins and then replace the feature with a discrete feature that takes
% value x if the original feature's value was in bin x.
% Input:
% k, number of bins
% attributes, a matrix of attributes and examples
% type_sizes, a column matrix indicating whether an attribute is nominal or
% continuous (0: continuous, otherwise is nominal)
% Output:
% discreteAttributes, a matrix where all nominal attributes in the original
% attributes matrix are preserved, and all continuous attributes in the
% original matrix are discretized
% Example:
% attributes=[1 2 3 4 5 6 7 8 9 10;1 2 1 2 1 2 1 2 1 2;10 20 30 40 50 60 70 80 90 100]';
% type_sizes=[0 2 0]';
% discAttr=kBins(3,attributes,type_sizes);
% The output is discAttr = [ 1 1 1 1 2 2 2 3 3 3; 1 2 1 2 1 2 1 2 1 2; 1 1 1 1 2 2 2 3 3 3]';
discreteAttributes=zeros(size(attributes));
% for each attribute
for i=1:length(type_sizes)
    bin=1:k;
    % if it is continuous
    if type_sizes(i)==0
        thisAttr=attributes(:,i);
        thisAttr=discretize(thisAttr,k,bin);
        discreteAttributes(:,i)=thisAttr;
    % else it is not continuous
    else    
        discreteAttributes(:,i)=attributes(:,i);
    end
end
end

function [logpos,logneg]=test(example,probmat)
% Subfunction: testing an example
% Test an example and output the natural log of probability of positive and negative
% Input:
% exmaple, an array that includes the value for each attribute
% probmat, a matrix that stores all probabilities needed for calculation
% Output:
% logpos,the natural log of the probability that this example is classified
% as positive given its values for each attribute
% logneg,the natural log of the probability that this example is classified
% as negative given its values for each attribute
logpos=0;
logneg=0;
l=length(example);
for i=1:l
    thisValue=example(i);
    posProbs=probmat(i+1,thisValue);
    if posProbs == 0
        logpos = 0 + logpos;
    else
        logpos=log(posProbs)+logpos;
    end
    negProbs=probmat(i+1,thisValue+length(probmat(1,:))/2);
    if negProbs == 0
        logneg = 0 + logneg;
    else
        logneg=log(negProbs)+logneg;
    end
end
if probmat(1,1) ~= 0
    logpos=logpos+log(probmat(1,1));
end
if probmat(1,2) ~= 0
    logneg=logneg+log(probmat(1,2));
end
end


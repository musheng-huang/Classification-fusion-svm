%%=====My Multi source domain generalizaiton method___ Multi--A======%%
%clear;clc;
close all


%%=====1. Load data=======%%
% load the domain A/B/C/D/E/F/G/H 
datapath = 'E:\Features4Bearing_4C\';
% 先不进行归一化
% 1. Domain A data
load([datapath,'CWRU_A_Features.mat'],'Features','Labels');
TrainS1 = Features;
A=Features;
LabelS1 = Labels;
clear Features Labels

% 2. Domain B data
load([datapath,'CWRU_B_Features.mat'],'Features','Labels');
TrainS2 = Features;
LabelS2 = Labels;
B=Features;
clear Features Labels

% 3. Domain C data
load([datapath,'CWRU_C_Features.mat'],'Features','Labels');
TrainS3 = Features;
LabelS3 = Labels;
C=Features;
clear Features Labels

% 4. Domain D data
load([datapath,'CWRU_D_Features.mat'],'Features','Labels');
TrainS4 = Features;
D=Features;
LabelS4 = Labels;
clear Features Labels

% 5. Domain G data
load([datapath,'CWRU_G_Features.mat'],'Features','Features_Normal4T','Labels');
TrainS7 = Features;
LabelS7 = Labels;
G=Features;
% TrainT = Features_Normal4T;
% LabelT = ones(1,length(TrainT(1,:)));

clear Features Labels Features_Normal4T


% ------ Normalize to [0,1]-------%
% 以Domain_B为基准对训练数据进行归一化，使得每一个样本的每一个特征处在[0,1]之间
[TrainS1,PS] = mapminmax(TrainS1,0,1);
% normalize the other domains to this scale
TrainS2 = mapminmax('apply',TrainS2,PS);
TrainS3 = mapminmax('apply',TrainS3,PS);
TrainS4 = mapminmax('apply',TrainS4,PS);

TrainS7 = mapminmax('apply',TrainS7,PS);
% TrainT = mapminmax('apply',TrainT,PS);

TestData = TrainS7;
TestLabel = LabelS7';

%%======Train multiple source karcher mean========%%
all_data = {TrainS1,TrainS2,TrainS3,TrainS4};
all_labels = {LabelS1,LabelS2,LabelS3,LabelS4};


md = 2:30;
Result = zeros(1,length(md));
Result1 = zeros(1,length(md));
Result3 = zeros(1,length(md));
Result2 = zeros(1,length(md));
Result4 = zeros(1,length(md));

md = 2:30;
Result = zeros(1,length(md));

for i = 1:length(md)
    r = md(i);    % the dimensionality of subspace
    k = 7;
    mtrans_out = multi_domains_gener(all_data, all_labels, r, k);

    %%======mapping TrainT and test data========%%
    mapped_trainS = mtrans_out.mapped_data;
    GM = mtrans_out.GM;
%     mapped_trainT = GM'*TrainT;
    mapped_TestData = TestData'*GM;
    TrainingData = [cell2mat(mapped_trainS)]';%,mapped_trainT
    TrainingLabel = [cell2mat(all_labels)]';%,LabelT
    
    %%======SVM classification model=======%%
    %%======SVM classification model=======%%
    bestc = 1;
    % RBF Kernel （应用RBF核函数实现2分类问题支持向量机的训练）
    options=['-s 0 -t 2 ',' -c ',num2str(bestc)];
     model_RBF1 = svmtrain(all_labels{1}',mapped_trainS{1}', options);
     model_RBF2 = svmtrain(all_labels{2}',mapped_trainS{2}', options);
      model_RBF3 = svmtrain(all_labels{3}',mapped_trainS{3}', options);
       model_RBF4 = svmtrain(all_labels{4}',mapped_trainS{4}', options);
           model_RBF = svmtrain(TrainingLabel,TrainingData, options);
        mmd_XY1=my_mmd(G,A,2);
           mmd_XY2=my_mmd(G,B,2);
           mmd_XY3=my_mmd(G,C,2);
           mmd_XY4=my_mmd(G,D,2);  
%             mmd_XY5=my_mmd(mapped_TestData,mapped_trainT',R);
           values=Untitled5(mmd_XY1,mmd_XY2,mmd_XY3,mmd_XY4,0.01);
%  [label1,score1] = predict(model_RBF1,mapped_TestData);
    [predict_label_L, accuracy_L, dec_values_L5] = svmpredict(TestLabel, mapped_TestData, model_RBF,'-b 0');
    [predict_label_L1, accuracy_L1, dec_values_L1] = svmpredict(TestLabel, mapped_TestData, model_RBF1,'-b 0');
    [predict_label_L2, accuracy_L2, dec_values_L2] = svmpredict(TestLabel, mapped_TestData, model_RBF2,'-b 0');
    [predict_label_L3, accuracy_L3, dec_values_L3] = svmpredict(TestLabel, mapped_TestData, model_RBF3,'-b 0');
    [predict_label_L4, accuracy_L4, dec_values_L4] = svmpredict(TestLabel, mapped_TestData, model_RBF4,'-b 0');
% 最终预测标签为k*1矩阵,k为预测样本的个数
% 求出测试样本在5个模型中预测为“正”得分的最大值，作为该测试样本的最终预测标签
score = [values(1).*dec_values_L1(:,6),values(2).*dec_values_L2(:,6),1.75*values(3).*dec_values_L3(:,6),1*values(4).*dec_values_L4(:,6)];
% 最终预测标签为k*1矩阵,k为预测样本的个数
predict_label=[predict_label_L1,predict_label_L2,predict_label_L3,predict_label_L4];
votes=vote(values,predict_label);
final_labels = zeros(1080,1);
for N = 1:size(final_labels,1)
    % 返回每一行的最大值和其位置
    [m,p] = max(score(N,:));
    % 位置即为标签
    final_labels(N,:) = p;
end
H=0;
Q=0;
group = TestLabel; % 真实标签
grouphat = final_labels; % 预测标签

for K=1:1080
    if group(K,1)==grouphat(K,1)
        H=H+1;
    end
end
for K=1:1080
    if group(K,1)==votes(K,1)
        Q=Q+1;
    end
end
    Result(1,i) = accuracy_L(1);
      Result1(1,i) = accuracy_L1(1);
        Result2(1,i) = accuracy_L2(1);
          Result3(1,i) = accuracy_L3(1);
            Result4(1,i) = accuracy_L4(1);
            R4T(1,i) = H/1080;
            R4T1(1,i)=Q/1080;

end
max(max(Result))
max(R4T1)
%  save (filename);

%  save (filename);
%  clear;clc;
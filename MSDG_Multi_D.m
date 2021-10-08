%%=====My Multi source domain generalizaiton method___ Multi--B======%%
% clear;clc;
close all
tic

%%=====1. Load data=======%%
% load the domain A/B/C/D/E/F/G/H 
datapath = 'E:\Features4Bearing_4C\';

% �Ȳ����й�һ��
% 1. Domain D data
load([datapath,'CWRU_D_Features.mat'],'Features','Features_Normal4T','Labels');
TrainS4 = Features;
LabelS4 = Labels;
D=Features;
% TrainT = Features_Normal4T;
% LabelT = ones(1,length(TrainT(1,:)));

clear Features Labels Features_Normal4T

% 5. Domain E data
load([datapath,'CWRU_E_Features.mat'],'Features','Labels');
TrainS5 = Features;
LabelS5 = Labels;
E=Features;
clear Features Labels

% 6. Domain F data
load([datapath,'CWRU_F_Features.mat'],'Features','Labels');
TrainS6 = Features;
LabelS6 = Labels;
F=Features;
clear Features Labels

% 7. Domain G data
load([datapath,'CWRU_G_Features.mat'],'Features','Labels');
TrainS7 = Features;
LabelS7 = Labels;
G=Features;
clear Features Labels

% 8. Domain H data
load([datapath,'CWRU_H_Features.mat'],'Features','Labels');
TrainS8 = Features;
H=Features;
LabelS8 = Labels;
clear Features Labels


% ------ Normalize to [0,1]-------%
% ��Domain_BΪ��׼��ѵ�����ݽ��й�һ����ʹ��ÿһ��������ÿһ����������[0,1]֮��
[TrainS5,PS] = mapminmax(TrainS5,0,1);
% normalize the other domains to this scale
TrainS4 = mapminmax('apply',TrainS4,PS);

TrainS6 = mapminmax('apply',TrainS6,PS);
TrainS7 = mapminmax('apply',TrainS7,PS);
TrainS8 = mapminmax('apply',TrainS8,PS);
% TrainT = mapminmax('apply',TrainT,PS);

TestData = TrainS4;
TestLabel = LabelS4';

%%======Train multiple source karcher mean========%%
all_data = {TrainS5,TrainS6,TrainS7,TrainS8};
all_labels = {LabelS5,LabelS6,LabelS7,LabelS8};


md = 2:30;
Result = zeros(1,length(md));
Result1 = zeros(1,length(md));
Result2 = zeros(1,length(md));
Result3 = zeros(1,length(md));
Result4 = zeros(1,length(md));

R4T = zeros(1,length(md));
R4T1 = zeros(1,length(md));

Result = zeros(1,length(md));

    U=0;
for i = 1:length(md)
      

          U=U+1;
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
    bestc = 1;
    % RBF Kernel ��Ӧ��RBF�˺���ʵ��2��������֧����������ѵ����
    options=['-s 0 -t 2 ',' -c ',num2str(bestc)];
     model_RBF1 = svmtrain(all_labels{1}',mapped_trainS{1}', options);
     model_RBF2 = svmtrain(all_labels{2}',mapped_trainS{2}', options);
     model_RBF3 = svmtrain(all_labels{3}',mapped_trainS{3}', options);
     model_RBF4 = svmtrain(all_labels{4}',mapped_trainS{4}', options);
           model_RBF = svmtrain(TrainingLabel,TrainingData, options);
        mmd_XY1=my_mmd(A,E,2);
           mmd_XY2=my_mmd(A,F,2);
           mmd_XY3=my_mmd(A,G,2);
           mmd_XY4=my_mmd(A,H,2);  
           values=Untitled5(mmd_XY1,mmd_XY2,mmd_XY3,mmd_XY4,MMD);
    [predict_label_L, accuracy_L, dec_values_L5] = svmpredict(TestLabel, mapped_TestData, model_RBF,'-b 0');
    [predict_label_L1, accuracy_L1, dec_values_L1] = svmpredict(TestLabel, mapped_TestData, model_RBF1,'-b 0');
    [predict_label_L2, accuracy_L2, dec_values_L2] = svmpredict(TestLabel, mapped_TestData, model_RBF2,'-b 0');
    [predict_label_L3, accuracy_L3, dec_values_L3] = svmpredict(TestLabel, mapped_TestData, model_RBF3,'-b 0');
    [predict_label_L4, accuracy_L4, dec_values_L4] = svmpredict(TestLabel, mapped_TestData, model_RBF4,'-b 0');
predict_label=[predict_label_L1,predict_label_L2,predict_label_L3,predict_label_L4];
votes=vote(values,predict_label);

Q=0;
group = TestLabel; % ��ʵ��ǩ

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
    
            R4T1(1,i)=Q/1080;

end
max(max(Result))
max(R4T1)

% %�����
% save (filename);
% toc
% disp(['����ʱ��: ',num2str(toc)]); 
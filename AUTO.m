clear;clc

name_num={0.075, 0.05, 0.025, 0.01,0.005};
R=3;
for n = 1:5
    filename=strcat('Aset2',num2str(n),'.mat');
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    MSDG_Multi_A
end
for n = 1:5
    filename=strcat('Bset2',num2str(n),'.mat');
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    MSDG_Multi_B
%     save (filename);
end
for n = 1:5
    filename=strcat('Cset2',num2str(n),'.mat');
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    MSDG_Multi_C
%     save (filename);
end
for n = 1:5
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    filename=strcat('Dset2',num2str(n),'.mat');
    
    MSDG_Multi_D
%     save (filename);
end
for n = 1:5
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    filename=strcat('Eset2',num2str(n),'.mat');
    MSDG_Multi_E
%     save (filename);
end
for n = 1:5
name_num={0.075, 0.05, 0.025, 0.01,0.005};
Z=name_num{n};
    filename=strcat('Fset2',num2str(n),'.mat');
    MSDG_Multi_F
%     save (filename);
end
for n = 1:5
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    filename=strcat('Gset2',num2str(n),'.mat');
    MSDG_Multi_G
%     save (filename);
end
for n = 1:5
name_num={0.075, 0.05, 0.025, 0.01,0.005};
    Z=name_num{n};
    filename=strcat('Hset2',num2str(n),'.mat');
    MSDG_Multi_H
%     save (filename);
end

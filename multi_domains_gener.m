function mtrans_out = multi_domains_gener(all_data, all_labels, r, k)
% This is the function for multi_source domains generalization method for
% intelligent fault identification
% using this function, each domain is mapped to grassmann manifold and the
% karcher mean is computed
% Input:
%      (1) all_data --- All training data of each domain, 1xm cell matrix
%                       where m is number of domain; For each domain, the
%                       data are organized by dxni matrix,where d is the
%                       dimension of each sample, ni is the sample size of
%                       domain i(i=1,...,m).
%      (2) all_labels -- All label information for each domain, 1xm cell matrix
%                       where m is number of domain; For each domain, the
%                       label data are organized by 1xni matrix,where ni is
%                       the sample size of domain i(i=1,...,m).
%      (3)   r -- the dimensionality of embedded space
%      (4)   k -- the neighbor num for LFDA method
% Output:
%      mtrans_out ---- the out structure contains the following fileds
%       (1) mtrans_out.S: the subspace of each domain
%       (2) mtrans_out.GM: the karcher mean of domains in grassmann manifold
%       (3) mtrans_out.mapped_data: the mapping data of each domain, 1xm
%              cell matraix, for each cell, it is a rxni matrix;
%       (4) mtrans_out.r: the dimensionality of subspace

%  References:
%      [1] Sugiyama M. Dimensionality reduction of multimodal labeled data by 
%          local fisher discriminant analysis[J]. Journal of machine learning research, 2007, 8(May): 1027-1061.
%      [2] Boumal N, Mishra B, Absil P A, et al. Manopt, a Matlab toolbox for
%          optimization on manifolds[J]. The Journal of Machine Learning Research, 2014, 15(1): 1455-1459.

% (C) Huailiang Zheng, Harbin institute of technology, 05/18, 2018;
%     hlzhenghit@126.com


if nargin < 3
    error('not enough input!')
end
if nargin < 7
    k = 7;
end

%%======1. LFDA subspaces=======%%
disp('Compute subspace for each domains...')
m = length(all_data);  % the number of domains
S = cell(1,m);
for i = 1:m
    [S{i},~] = LFDA(all_data{i},all_labels{i}',r,'orthonormalized',k); 
end

%%======2. Karcher Mean on Grassmann Manifold======%%
disp('Compute Karcher mean ...')
d = size(S{1},1);
A = zeros(d, r, m);
for i = 1:m
    A(:,:,i) = S{i};
end
GM = grassmann_karcher_mean(A);

%%=====3. Mapping training data=======%%
mapped_data = cell(1,m);
for i = 1:m
    mapped_data{i} = GM'*all_data{i};
end

%%=====4. Store the output======%%
mtrans_out.S = S;
mtrans_out.GM = GM;
mtrans_out.mapped_data = mapped_data;

  
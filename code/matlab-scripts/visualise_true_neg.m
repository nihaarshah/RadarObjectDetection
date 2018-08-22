% Visualise several figures of the true positives
load('/Users/nihaar/Documents/4yp/data/classification-mistakes/SVM-aligned_without-hnm/TN/true_neg_hard.mat','tn_hard_mat');
% load('fake_features_matrix');
tn_mat = tn_hard_mat;
[m,n] = size(tn_mat);
figure(); hold on
pause(10);
for i = 1:m
    title('True Negative Hard')
    img = reshape(tn_mat(i,:),[41,41]);
    imagesc(img);
    hold on;
    pause(0.5);
end
hold off;
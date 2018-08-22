% Visualise several figures of the false positives
load('/Users/nihaar/Documents/4yp/data/classification-mistakes/SVM-aligned_without-hnm/false_positives_svm.mat','fp_mat');
% load('fake_features_matrix');
fp_mat = fp_mat;
[m,n] = size(fp_mat);
figure(); hold on
pause(10);
for i = 1:m
    title('False Positives')
    img = reshape(fp_mat(i,:),[41,41]);
    imagesc(img);
    hold on;
    pause(0.5);
end
hold off;
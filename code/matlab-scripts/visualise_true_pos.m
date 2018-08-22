% Visualise several figures of the true positives
load('/Users/nihaar/Documents/4yp/data/classification-mistakes/SVM-aligned_without-hnm/true_pos_hard.mat','tp_hard_mat');
% load('fake_features_matrix');
tp_mat = tp_hard_mat;
[m,n] = size(tp_mat);
figure(); hold on
pause(10);
for i = 1:m
    title('True Positives Hard')
    img = reshape(tp_mat(i,:),[41,41]);
    imagesc(img);
    hold on;
    pause(0.5);
end
hold off;
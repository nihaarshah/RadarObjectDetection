% Visualise several figures of the false positives
load('/Users/nihaar/Documents/4yp/data/classification-mistakes/SVM-aligned_without-hnm/false_negatives_svm.mat','fn_mat');
[m,n] = size(fn_mat);
figure(); hold on
pause(10);
for i = 1:m
    title('False Negatives')
    img = reshape(fn_mat(i,:),[41,41]);
    imagesc(img);
    hold on;
    pause(0.8);
end
hold off;
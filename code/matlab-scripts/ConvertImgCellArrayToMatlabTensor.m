% This script converst a matlab 1 x N cell containing N image patches into
% a matlab 3d matrix/tensor (N, Image_height, Image_width) N images stacked



% Doing for positive examples
load('~/Documents/4yp-code/data/taxi-rank2-car-patches/XTrainImagesPos.mat')
N = length(XTrainImagesPos);
XTrainPos = zeros(N,39,39);
XTrainTrimmedPos = zeros(N,36,36);

for i = 1:N
XTrainPos(i,:,:) = XTrainImagesPos{i};
XTrainTrimmedPos(i,:,:) = XTrainPos(i,2:37,3:38);
end
save('XTrainTrimmedPos','XTrainTrimmedPos');


% Doing for negative examples
load('~/Documents/4yp-code/data/taxi-rank2-car-patches/XTrainImagesNeg.mat')
N = length(XTrainImagesNeg);
XTrainNeg = zeros(N,39,39);
XTrainTrimmedNeg = zeros(N,36,36);

for i = 1:N
XTrainNeg(i,:,:) = XTrainImagesNeg{i};
XTrainTrimmedNeg(i,:,:) = XTrainNeg(i,2:37,3:38);
end
save('XTrainTrimmedNeg','XTrainTrimmedNeg');



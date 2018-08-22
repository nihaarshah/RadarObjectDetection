% This function thresholds each patch to reduce noise on patches
function [filtered_patch] = ThresholdFilterReduceSpeckleNoise(UnfilteredPatch)

x = UnfilteredPatch;
indices = find(x<=0.30);
x(indices) = 1e-3;


filtered_patch = x;
end
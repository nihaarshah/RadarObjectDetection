
flc = frame_labels_cell;

frames = length(flc);
gt_cell = cell(1,frames);

for i = 1:frames
    A = flc{i};
    C = [];
    for j = 1:size(A,2)
        B = [A(1,j),A(2,j),A(1,j)+36,A(2,j)+36];
        B = round(B);
        C = [C;B];
        
    end
    gt_cell{i} = C;
end
        
   
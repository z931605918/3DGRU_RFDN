clear;
filepath = 'D:\\5.dataset\\范_圆柱绕流CFD\\流场flo截取'; %filepath是原始数据的位置

mat_path=dir(filepath);
u=[];
v=[];
path = 'D:\\5.dataset\\范_圆柱绕流CFD\\LR数据4倍'; %生成数据集的路径，
for i=3:length(mat_path)
    mat_name=mat_path(i).name;
   
    img1 = readFlowFile(strcat(mat_path(i).folder,'/',mat_name));
    ImLR = imresize_BD(img1,4, [16,16], 5); %第二个数改采样倍数
    u = ImLR(:,:,1);
    v = ImLR(:,:,2);
    h=size(u);
    mat_name = mat_name(1:length(mat_name)-14);
    flo_name=mat_name;
    writeFlowFile(ImLR, strcat(path,'\',flo_name,num2str(i-2,'_%04d'),'.flo'));
    %writeFlowFile(ImLR, strcat(path,'\','backstep_',num2str(i-2,'%03d'),'.flo'));
    
  
end

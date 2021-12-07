clc;clear;
tic;
I=imread('cameraman.tif');
[idx,idy]=size(I);
length=idx*idy;
N=length;%
K=256;%k个种子点，聚类中心
m=30;%调节系数
S=floor(sqrt(N/K));%大像素之间的距离
NewCluster=zeros(K,4);
% seeds=zeros(K,3);
%% 初始化聚类中心
% SeedVector=zeros(K,2);%存放种子点的坐标
XOffset=0;
YOffset=0;
n=floor(sqrt(K));
SeedVectorOld = zeros(K, 3);
for j=1:1:n
    for i=1:1:n
        SeedVectorOld(i+n*(j-1),1)=floor(S/2)+XOffset;
        SeedVectorOld(i+n*(j-1),2)=floor(S/2)+YOffset;
        YOffset=YOffset+S;
    end
    XOffset=XOffset+S;
    YOffset=0;
end 
%% 3*3邻域选择最小梯度
G=I;
SeedVector=SeedVectorOld;

for i=1:1:idx-1
   for j=1:1:idy-1
        dx=I(i+1,j)-I(i,j);
        dy=I(i,j+1)-I(i,j);
        G(i,j)=dx+dy;
   end
end
%遍历所有种子点，移动到邻域内的最小的点的位置
for i=1:1:K
   Gx=SeedVector(i,1);
   Gy=SeedVector(i,2);
   A=G(Gx-1:Gx+1, Gy-1:Gy+1);
   [Xindex,Yindex]=find(A==min(min(A)));
   SeedVector(i,1)=SeedVector(i,1)+Xindex(1)-2;
   SeedVector(i,2)=SeedVector(i,2)+Yindex(1)-2;
end
seeds=SeedVector;
%% 
Labels = -1*ones(idx,idx);
Distance = inf(idx,idy);
error=0.0001;
residual=1;
I=double(I);
iternum = 1;
iters=1000;
% while residual>error
while iternum<iters
    for k=1:1:K
        sx=floor(seeds(k,1));
        sy=floor(seeds(k,2));
        % 2S*2S
        for i=max(1, floor(sx-floor(S))+1):1:min(idx, floor(sx+floor(S)))
            for j=max(1, floor(sy-floor(S))+1):1:min(idy, floor(sy+floor(S)))
                dc=sqrt((I(i,j)-I(sx,sy))^2);
                ds=sqrt((i-sx)^2+(j-sy)^2);
                D= sqrt(dc*dc+m^2*(ds/S)^2);
                if D<Distance(i,j)
                   Distance(i,j) = D;
                   Labels(i,j) = k;
                end
            end
        end
    end
    
    %计算新的聚类中心
    NewCluster=zeros(K,4);
     for ix = 1:idx
            for iy = 1:idy
                label = Labels(ix,iy);
                if(label==-1)
                    continue
                end
%                 disp([label, ix, iy])
                %将当前类别号在x,y点的I,x,y值进行累加至NewCluster
                NewCluster(label,1) = NewCluster(label,1)+ix;
                NewCluster(label,2) = NewCluster(label,2)+iy;
                NewCluster(label,3) = NewCluster(label,3)+I(ix,iy);
                NewCluster(label,4) = NewCluster(label,4)+1;
            end
     end
        SeedsOld = seeds;
        %求平均，重新计算该种子点的中心点
        for i = 1:K
            seeds(i,1:3) = round(NewCluster(i,1:3)/NewCluster(i,4));
        end
%         %判断是否收敛
        curErr = norm(norm(SeedsOld-seeds))
        if curErr<error
            break;
        end
    iternum=iternum+1;
end
disp(iternum)
%% 增强连通性
S_Search=16;%合并孤立点的筛选面积
%将小区域合并
LabelsOld = Labels;
    for i = 1:K
        %寻找第i标签superpixels标注
        emptylabels = zeros(idx,idy);
        emptylabels(Labels == i) = 1;
        [L_tmp,num] = bwlabel(emptylabels,4);
        for j=1:num
            if sum(sum(L_tmp==j))<S_Search
                [x,y]=find(L_tmp==j);
                A = [];
                for ix=1:1:size(x)
                     up=Labels(x(ix),min(y(ix)+1,idy));
                     down=Labels(x(ix),max(y(ix)-1,1));
                     right=Labels(min(x(ix)+1,idx),y(ix));
                     left=Labels(max(x(ix)-1,1),y(ix));
                     if up~=Labels(x(ix),y(ix))
                         A = [A, up];
                     end
                     if down~=Labels(x(ix),y(ix))
                         A = [A, down];
                     end
                     if left~=Labels(x(ix),y(ix))
                         A = [A, left];
                     end
                     if right~=Labels(x(ix),y(ix))
                         A = [A, right];
                     end
                end
                if size(A)~=0
                    Labels(x,y) = mode(A);
                end
            end
        end 
    end
%% 绘图
% subplot(121)
% imshow(I,[]);title('原图');
% subplot(122)
imshow(I,[]);hold on;
% plot(seeds(:,1),seeds(:,2),'.r');hold on;
contour(Labels,K,'b');
mstr=num2str(m);
Kstr=num2str(K);
str=['SLIC m=',mstr,' 超像素个数K=',Kstr];
title(str);
saveas(gca,'SLIC_1024.tif');
toc

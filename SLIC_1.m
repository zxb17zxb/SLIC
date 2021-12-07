clc;clear;
tic;
I=imread('cameraman.tif');
[idx,idy]=size(I);
length=idx*idy;
N=length;%
K=256;%k�����ӵ㣬��������
m=30;%����ϵ��
S=floor(sqrt(N/K));%������֮��ľ���
NewCluster=zeros(K,4);
% seeds=zeros(K,3);
%% ��ʼ����������
% SeedVector=zeros(K,2);%������ӵ������
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
%% 3*3����ѡ����С�ݶ�
G=I;
SeedVector=SeedVectorOld;

for i=1:1:idx-1
   for j=1:1:idy-1
        dx=I(i+1,j)-I(i,j);
        dy=I(i,j+1)-I(i,j);
        G(i,j)=dx+dy;
   end
end
%�����������ӵ㣬�ƶ��������ڵ���С�ĵ��λ��
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
    
    %�����µľ�������
    NewCluster=zeros(K,4);
     for ix = 1:idx
            for iy = 1:idy
                label = Labels(ix,iy);
                if(label==-1)
                    continue
                end
%                 disp([label, ix, iy])
                %����ǰ������x,y���I,x,yֵ�����ۼ���NewCluster
                NewCluster(label,1) = NewCluster(label,1)+ix;
                NewCluster(label,2) = NewCluster(label,2)+iy;
                NewCluster(label,3) = NewCluster(label,3)+I(ix,iy);
                NewCluster(label,4) = NewCluster(label,4)+1;
            end
     end
        SeedsOld = seeds;
        %��ƽ�������¼�������ӵ�����ĵ�
        for i = 1:K
            seeds(i,1:3) = round(NewCluster(i,1:3)/NewCluster(i,4));
        end
%         %�ж��Ƿ�����
        curErr = norm(norm(SeedsOld-seeds))
        if curErr<error
            break;
        end
    iternum=iternum+1;
end
disp(iternum)
%% ��ǿ��ͨ��
S_Search=16;%�ϲ��������ɸѡ���
%��С����ϲ�
LabelsOld = Labels;
    for i = 1:K
        %Ѱ�ҵ�i��ǩsuperpixels��ע
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
%% ��ͼ
% subplot(121)
% imshow(I,[]);title('ԭͼ');
% subplot(122)
imshow(I,[]);hold on;
% plot(seeds(:,1),seeds(:,2),'.r');hold on;
contour(Labels,K,'b');
mstr=num2str(m);
Kstr=num2str(K);
str=['SLIC m=',mstr,' �����ظ���K=',Kstr];
title(str);
saveas(gca,'SLIC_1024.tif');
toc

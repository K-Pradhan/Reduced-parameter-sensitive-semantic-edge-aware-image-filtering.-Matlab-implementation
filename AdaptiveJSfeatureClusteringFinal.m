function [FilteredImage] = AdaptiveJSfeatureClusteringFinal(InputImage,Wjsd,Model)
% clear; clc; close all;

seg_lmt=5; Flag=Model; wlen = 3;
% Ref_Img = im2double(InputImage);
% disp('Size of the Image');
[m,n,c] = size(InputImage);

if(c>1)
GrayImg = rgb2gray(InputImage);
else
GrayImg = InputImage;
end
GrayImg = im2double(GrayImg); 

nbd = ones(3); 
GrayImg1 = imopen(InputImage,nbd); GrayImg2 = imclose(InputImage,nbd); 
InputForEdgeMap = ((1*InputImage+1*GrayImg1+1*GrayImg2)/3); 
NoOfFtr = 8; modelFlg = Flag; wf = Wjsd-1; wmin = 3;
for rpt =1:1
Fibo = [1 1 2 3 5 8 13 21 34 55]; % wnid = 0; 
  
eps = 0.001; 
JSDF = []; wi = 0;
for w=Fibo(([wmin:wmin+wlen])+wf)
    wi = wi+1;  
%     JSDF1 = []; JSDF2 = [];
for band =1:c
    GrayImg = InputForEdgeMap(:,:,band); 
    GrayImg = im2double(GrayImg); 
JSDtemp1 = zeros(m*n,NoOfFtr/2); 
AW = 1:w;
%  w=w+1;
[X,Y] = meshgrid(-w:w,-w:w);
W = -w:w;
parfor pxl =  1:m*n
[i,j] = ind2sub([m,n],pxl);
wm = w;
NbrH = pxl + W*m;  NbrH(NbrH<=0 | NbrH>m*n)= pxl; MDH = abs(Gmean(GrayImg(NbrH(wm:-1:1)))-Gmean(GrayImg(NbrH(wm+2:end))))+eps;%Entrpy(GrayImg(NbrH));%/sum(Pxl_Range(NbrH).^2);
NbrV = pxl + W;  NbrV(NbrV<=0 | NbrV>m*n)= pxl; MDV = abs(Gmean(GrayImg(NbrV(wm:-1:1)))-Gmean(GrayImg(NbrV(wm+2:end))))+eps;%Entrpy(GrayImg(NbrV));%/sum(Pxl_Range(NbrV).^2);
NbrD1 = pxl + W*m + W;  NbrD1(NbrD1<=0 | NbrD1>m*n)= pxl; MDD1 = abs(Gmean(GrayImg(NbrD1(wm:-1:1)))-Gmean(GrayImg(NbrD1(wm+2:end))))+eps;%Entrpy(GrayImg(NbrD1));%/sum(Pxl_Range(NbrD1).^2);
NbrD2 = pxl + W*m - W;  NbrD2(NbrD2<=0 | NbrD2>m*n)= pxl; MDD2 = abs(Gmean(GrayImg(NbrD2(wm:-1:1)))-Gmean(GrayImg(NbrD2(wm+2:end))))+eps;%Entrpy(GrayImg(NbrD2));%/sum(Pxl_Range(NbrD2).^2);

LocalPatternH =  abs(GrayImg(NbrH) - min([mean(GrayImg(NbrH(1:w))),mean(GrayImg(NbrH(w+2:2*w+1)))]))+ eps; % LocalPattern = LocalPattern(:,3); abs(GrayImg(NbrH)-GrayImg(pxl)); %
LocalPatternH = [LocalPatternH(1:w) LocalPatternH(w+2:end)]; % LocalPatternH = [GrayImg(NbrH(1:w)) GrayImg(NbrH(w+2:end))] >= (max(GrayImg(ImdNbrH(1:end)))+min(GrayImg(ImdNbrH(1:end))))/2;% % LocalPatternH = LocalPatternH >= (mean(LocalPatternH(1:w))+mean(LocalPatternH(w+1:end)))/2; %% LocalPatternH < median(GrayImg(ImdNbr(1:end)));% (max(LocalPatternH)+min(LocalPatternH))/2; % mean(LocalPatternH);
LocalPatternHR =  abs(GrayImg(NbrH) - max([mean(GrayImg(NbrH(1:w))),mean(GrayImg(NbrH(w+2:2*w+1)))]))+ eps;
LocalPatternHR = [LocalPatternHR(1:w) LocalPatternHR(w+2:end)]; % LocalPatternHR = [GrayImg(NbrH(1:w)) GrayImg(NbrH(w+2:end))] <= (max(GrayImg(ImdNbrH(1:end)))+min(GrayImg(ImdNbrH(1:end))))/2;%

if(mean(LocalPatternH(1:w))<=mean(LocalPatternH(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
% size(GrayImg(NbrH))
% LocalPatternH = GrayImg(NbrH)>= Rmid;
JSDH1 = JSDiv(LocalPatternH,StandaredPattern); 

if(mean(LocalPatternHR(1:w))<=mean(LocalPatternHR(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
% JSDH1R = JSDiv(LocalPatternHR,StandaredPattern);
StandaredPattern =   ones(1,2*w);%/length(LocalPattern); GrayImg(NbrH);%
JSDH2 = JSDiv(LocalPatternH,StandaredPattern); JSDH2R = JSDiv(LocalPatternHR,StandaredPattern);

% ImdNbrV = [NbrV];% NbrD1 NbrD2]; % ImdNbr; %
LocalPatternV =  abs(GrayImg(NbrV) - min([mean(GrayImg(NbrV(1:w))),mean(GrayImg(NbrV(w+2:2*w+1)))]))+ eps; % LocalPattern = LocalPattern(:,3); abs(GrayImg(NbrV)-GrayImg(pxl)); %
LocalPatternV = [LocalPatternV(1:w) LocalPatternV(w+2:end)]; % LocalPatternV = [GrayImg(NbrV(1:w)) GrayImg(NbrV(w+2:end))] >= (max(GrayImg(ImdNbrV(1:end)))+min(GrayImg(ImdNbrV(1:end))))/2;% % LocalPatternV = LocalPatternV >= (mean(LocalPatternV(1:w))+mean(LocalPatternV(w+1:end)))/2; %%median(LocalPatternV);%(max(LocalPatternV)+min(LocalPatternV))/2; %mean(LocalPatternV);
LocalPatternVR =  abs(GrayImg(NbrV) - max([mean(GrayImg(NbrV(1:w))),mean(GrayImg(NbrV(w+2:2*w+1)))]))+ eps;
LocalPatternVR = [LocalPatternVR(1:w) LocalPatternVR(w+2:end)]; % LocalPatternVR = [GrayImg(NbrV(1:w)) GrayImg(NbrV(w+2:end))] <= (max(GrayImg(ImdNbrV(1:end)))+min(GrayImg(ImdNbrV(1:end))))/2;%

if(mean(LocalPatternV(1:w))<=mean(LocalPatternV(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
JSDV1 = JSDiv(LocalPatternV,StandaredPattern);

if(mean(LocalPatternVR(1:w))<=mean(LocalPatternVR(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
% JSDV1R = JSDiv(LocalPatternVR,StandaredPattern);
StandaredPattern =  ones(1,2*w);%/length(LocalPattern); GrayImg(NbrV);%
JSDV2 = JSDiv(LocalPatternV,StandaredPattern); JSDV2R = JSDiv(LocalPatternVR,StandaredPattern);

% ImdNbrD1 = [NbrD1]; % ImdNbr; % NbrH NbrV 
LocalPatternD1 =  abs(GrayImg(NbrD1) - min([mean(GrayImg(NbrD1(1:w))),mean(GrayImg(NbrD1(w+2:2*w+1)))]))+ eps; % LocalPattern = LocalPattern(:,3); abs(GrayImg(NbrD1)-GrayImg(pxl)); %

LocalPatternD1 = [LocalPatternD1(1:w) LocalPatternD1(w+2:end)]; % LocalPatternD1 = [GrayImg(NbrD1(1:w)) GrayImg(NbrD1(w+2:end))] >= (max(GrayImg(ImdNbrD1(1:end)))+min(GrayImg(ImdNbrD1(1:end))))/2;%% LocalPatternD1 = LocalPatternD1 >= (mean(LocalPatternD1(1:w))+mean(LocalPatternD1(w+1:end)))/2; %%median(LocalPatternD1);%(max(LocalPatternD1)+min(LocalPatternD1))/2; % mean(LocalPatternD1);
LocalPatternD1R =  abs(GrayImg(NbrD1) - max([mean(GrayImg(NbrD1(1:w))),mean(GrayImg(NbrD1(w+2:2*w+1)))]))+ eps;
LocalPatternD1R = [LocalPatternD1R(1:w) LocalPatternD1R(w+2:end)]; % LocalPatternD1R = [GrayImg(NbrD1(1:w)) GrayImg(NbrD1(w+2:end))] <= (max(GrayImg(ImdNbrD1(1:end)))+min(GrayImg(ImdNbrD1(1:end))))/2;%

if(mean(LocalPatternD1(1:w))<=mean(LocalPatternD1(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
JSDD11 = JSDiv(LocalPatternD1,StandaredPattern);

if(mean(LocalPatternD1R(1:w))<=mean(LocalPatternD1R(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
% JSDD11R = JSDiv(LocalPatternD1R,StandaredPattern);
StandaredPattern =   ones(1,2*w);%/length(LocalPattern);  imcomplement(StandaredPattern); % GrayImg(NbrD1);%
JSDD12 = JSDiv(LocalPatternD1,StandaredPattern); JSDD12R = JSDiv(LocalPatternD1R,StandaredPattern);


LocalPatternD2 =  abs(GrayImg(NbrD2) - min([mean(GrayImg(NbrD2(1:w))),mean(GrayImg(NbrD2(w+2:2*w+1)))]))+ eps; % LocalPattern = LocalPattern(:,3); abs(GrayImg(NbrD2)-GrayImg(pxl)); %
% LocalPatternD2 = (LocalPatternD2-min(LocalPatternD2))./(max(LocalPatternD2)-min(LocalPatternD2)) + eps;
% LocalPatternD2 = GrayImg1(NbrD2);
LocalPatternD2 = [LocalPatternD2(1:w) LocalPatternD2(w+2:end)]; % LocalPatternD2 = [GrayImg(NbrD1(1:w)) GrayImg(NbrD1(w+2:end))] >= (max(GrayImg(ImdNbrD2(1:end)))+min(GrayImg(ImdNbrD2(1:end))))/2;% LocalPatternD2 = LocalPatternD2 >= (mean(LocalPatternD2(1:w))+mean(LocalPatternD2(w+1:end)))/2; %% median(LocalPatternD2);% (max(LocalPatternD1)+min(LocalPatternD1))/2; % mean(LocalPatternD2);
LocalPatternD2R =  abs(GrayImg(NbrD2) - max([mean(GrayImg(NbrD2(1:w))),mean(GrayImg(NbrD2(w+2:2*w+1)))]))+ eps;
% LocalPatternD2R = (LocalPatternD2R-min(LocalPatternD2R))./(max(LocalPatternD2R)-min(LocalPatternD2R)) + eps;
LocalPatternD2R = [LocalPatternD2R(1:w) LocalPatternD2R(w+2:end)]; % LocalPatternD2R = [GrayImg(NbrD2(1:w)) GrayImg(NbrD2(w+2:end))] <= (max(GrayImg(ImdNbrD2(1:end)))+min(GrayImg(ImdNbrD2(1:end))))/2;%

if(mean(LocalPatternD2(1:w))<=mean(LocalPatternD2(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end  %  + eps/255;
% LocalPatternD2 = GrayImg(NbrD2)>= Rmid;
JSDD21 = JSDiv(LocalPatternD2,StandaredPattern); 

if(mean(LocalPatternD2R(1:w))<=mean(LocalPatternD2R(w+1:end)))
 StandaredPattern = [zeros(1,w) ones(1,w)];% + eps/255
else
 StandaredPattern = [ones(1,w) zeros(1,w)]; 
end
% JSDD21R = JSDiv(LocalPatternD2R,StandaredPattern);
StandaredPattern =   ones(1,2*w);%/length(LocalPattern);  imcomplement(StandaredPattern); % GrayImg(NbrD1);%
JSDD22 = JSDiv(LocalPatternD2,StandaredPattern); JSDD22R = JSDiv(LocalPatternD2R,StandaredPattern);

temp1 = [JSDH1 JSDV1 JSDD11 JSDD21];
% temp1R = [JSDH1R JSDV1R JSDD11R JSDD21R];

tempS = temp1; % min([temp1;temp1R],[],1); %  min([temp1,[],1]); %

temp2 = [JSDH2 JSDV2 JSDD12 JSDD22]; 
% temp2R = [JSDH2R JSDV2R JSDD12R JSDD22R];
% temp =  [temp2;temp2R]; % tempU1 = abs(temp2-temp2R); temp2R; %
tempU =  temp2; %max(temp,[],1); % temp; %

tempMD = [MDH MDV MDD1 MDD2]; 

if (modelFlg == 2)
    tempMD = (tempS).*(1+exp(-(tempMD+tempU)));
    JSDtemp1(pxl,:) = tempMD.*([(tempS)>mean(tempU)]);
%     tempMD = (tempU+tempMD).*((1+exp(-(tempS))));  
%     JSDtemp2(pxl,:) = tempMD.*([(tempS)<=mean(tempU)]);
elseif (modelFlg == 1)
    tempMD = (tempU+tempMD).*((1+exp(-(tempS)))); 
    JSDtemp1(pxl,:) = tempMD.*([(tempS)<=mean(tempU)]);
%     tempMD = (tempS).*(1+exp(-(tempMD+tempU)));   
%     JSDtemp2(pxl,:) = tempMD.*([(tempS)>mean(tempU)]);
end
end
end
end
JSDF = [JSDF JSDtemp1];
JSDF =  reshape(JSDF,m*n,[]);
%[idx,Cntrd] =  kmeans(JSDF,2);
% Perform fuzzy C-means clustering
[Cntrd, U, ~] = fcm(JSDF, 2);
% To get hard cluster assignments (like idx in kmeans):
[~, idx] = max(U);  % idx will be 1xN vector of cluster indices
% [idx,Cntrd] = kmedoids(JSDF,2);
BW1 = idx == 1;
% figure;
% imshow(mat2gray((reshape(BW1,[m,n]))));
BW2 = idx == 2;
% figure;
% imshow(mat2gray((reshape(BW2,[m,n]))));

GrayImg = GrayImg1;
if (nnz(BW1)<=nnz(BW2)) 
    BW = BW1;
else
    BW = BW2;
end

%#########################################
w=1; BW_temp = BW;
[X,Y] = meshgrid(-w:w,-w:w);
W = -w:w;
figure;
imshow(mat2gray((reshape(BW,[m,n]))));
%#########################################

% Ref_Img = im2double(InputImage);
% lmt = 3;
for rpt= 1:seg_lmt 
FilteredImage=InputImage;
[FilteredImage] = AWRMedFilter(FilteredImage,BW,GrayImg); % FinalAdaptiveFilter(FilteredImage,BW,GrayImg);

% figure;
% imshow(mat2gray(FilteredImage));  % halt;
% 
% figure;
% imshow(mat2gray(2*InputImage-FilteredImage));

InputImage=FilteredImage;
end
end

% ############### Texture Structure Decomposition ###############
% figure;
% imshow(mat2gray(2*Ref_Img-FilteredImage));  % halt;
% % InputImage = localtonemap (InputImage);
% figure;
% imshow(FilteredImage); 
% InputImg = (((Ref_Img)-FilteredImage));
% figure;
% imshow(InputImg); 
% figure;
% imshow(exp((FilteredImage-max(max(max(FilteredImage))))*1/2+InputImg));
% InputImage = exp((FilteredImage-max(max(max(FilteredImage))))*1/2+InputImg);
% end
% InputImage = InputImage-min(min(min(InputImage)))/(max(max(max(InputImage)))-min(min(min(InputImage))));
% figure;
% imshow(InputImage);
% InputImage = histeq(InputImage);
% figure;
% imshow(InputImage);
% #################################### %

% %%$$$$$$$$$$$$$$$$$$$$$$$$$$$$%%
% FilteredImage = im2double(FilteredImage); InputImage = Ref_Img;
% [peaksnr, snr] = psnr(FilteredImage,Ref_Img)
% mse = immse(FilteredImage,Ref_Img)
% ssim = ssim(FilteredImage,Ref_Img)
% ESSIM = ESSIM(Ref_Img,FilteredImage)
% EPI = EPI(FilteredImage,Ref_Img)
% piqeO = piqe(Ref_Img)
% piqeF = piqe(FilteredImage)
% multissim = multissim(FilteredImage,Ref_Img)
% %%############################%%
% EI=[];EF=[];MI=[];JE=[];KLD=[]; JSD=[];
% for cb=1:3
% data_x = InputImage(:,:,cb); % data_x = data_x(1:m1*n1);
% data_y = FilteredImage(:,:,cb); % data_y = data_y(1:m2*n2);
% % N = 256;
% [p_x,p_y,p_xy] = jointhist(data_x,data_y); % p_xy = histcounts2(data_x,data_y,N); p_x = histcounts(data_x,N); p_y = histcounts(data_y,N);
% % sum(p_x)
% % sum(p_y)
% % sum(sum(p_xy)) % p_xy(p_xy == -inf & p_xy == -inf)=0; % p_xy = p_xy/sum(sum(p_xy));
% log2pxy = log2(p_xy);  log2pxy(log2pxy == -inf | log2pxy == inf)=0;
% % p_x(p_x == -inf & p_x == -inf)=0; % p_x = p_x/(sum(p_x));  
% % p_y(p_y == -inf & p_y == -inf)=0; % p_y = p_y/(sum(p_y));  
% log2px = log2(p_x); log2px(log2px == -inf | log2px == inf)=0;
% log2py = log2(p_y); log2py(log2py == -inf | log2py == inf)=0;
% px_y = p_x/p_y; px_y(px_y == -inf | px_y == inf)=0; log2px_y = log2(px_y);% px_y = px_y/(sum(px_y));
% py_x = p_y/p_x; py_x(py_x == -inf | py_x == inf)=0; log2py_x = log2(py_x); % py_x = py_x/(sum(py_x));
% log2px_y(log2px_y == -inf | log2px_y == inf)=0; log2py_x(log2py_x == -inf | log2py_x == inf)=0;
% p_x_y = (p_x+p_y)/2; p_x_yx = p_x_y/p_x; p_x_yy = p_x_y/p_y;
% p_x_yx(p_x_yx == -inf | p_x_yx == inf)=0; p_x_yy(p_x_yy == -inf | p_x_yy == inf)=0;
% log2p_x_yx = log2(p_x_yx); log2p_x_yx(log2p_x_yx == -inf | log2p_x_yx == inf)=0; % p_x_y = p_x_y/(sum(p_x_y)); 
% log2p_x_yy = log2(p_x_yy); log2p_x_yy(log2p_x_yy == -inf | log2p_x_yy == inf)=0;
% % p_x = p_x/sum(sum(p_x)); p_y = p_y/sum(sum(p_y)); 
% EI = [EI sum(-p_x.*log2px)];
% EF = [EF sum(-p_y.*log2py)];
% JE = [JE sum(sum(-p_xy.*log2pxy))];
% MI = [MI sum(-p_x.*log2px) + sum(-p_y.*log2py) - sum(sum(-p_xy.*log2pxy))];
% KLD = [KLD (sum(-p_x.*log2py_x) + sum(-p_y.*log2px_y))];
% JSD = [JSD (sum(-p_x.*log2p_x_yx) + sum(-p_y.*log2p_x_yy))/2];
% end

% MeanMI = mean(MI)

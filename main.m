% clear; clc; close all;
Wjsd = 1; Model = 2; % Wjsd = 1/2/3/4 and Model= 1/2

[InputImage] = imread('InputImages/02.jpg');
figure;
imshow(InputImage);
% InputImage = imnoise(InputImage,'gaussian',0,0.05);
InputImage = im2double(InputImage);
[FilteredImage] = AdaptiveJSfeatureClusteringFinal(InputImage,Wjsd,Model);

figure;
imshow(FilteredImage);
clc;
close all;

% STYLING
% Camel Case for Variable Names
% Underscored for Constant Parameters

% ----------------------- 0 IMAGE READING GUI ----------------------
[FILENAME, PATHNAME] = uigetfile('*.jpg', 'Select the Lesion Image');
filePath = strcat(PATHNAME,FILENAME);
disp('The image file location you selected is:')
disp(filePath);
% Read in image
im = imread(filePath);
figure,imshow(im);
title('0. Input Image:');

asSnakeSegmentationOut(im)
function lesionMask = asSnakeSegmentation(image, numIterations)
% asSnakeSegmentation - Adaptive Snake (AS) segmentation using user-defined mask
% Inputs:
%   image - RGB dermoscopic image
%   numIterations - Number of iterations for the snake (e.g., 70)
% Output:
%   lesionMask - Binary mask of the segmented lesion

if nargin < 2
    numIterations = 1000;
end

% Convert image to grayscale
grayImg = rgb2gray(im2double(image));

% User draws rough lesion outline
figure, imshow(image); title('Draw a rough outline around the lesion');
roi = drawfreehand();
wait(roi);  % Wait for user to finish
initMask = createMask(roi);

% Show the initial mask
figure; imshow(initMask); title('Initial User-Drawn Mask');

% Run active contour using Chan-Vese model
lesionMask = activecontour(grayImg, initMask, numIterations);

% Show final result
figure; imshow(labeloverlay(image, lesionMask));
title(sprintf('Chan-Vese Segmentation (Iterations = %d)', numIterations));

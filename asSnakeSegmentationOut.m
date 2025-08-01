function lesionMask = asSnakeSegmentationOut(image, numIterations)
% asSnakeSegmentation - Shrinking Snake Segmentation using user-defined outer mask
% Inputs:
%   image - RGB dermoscopic image
%   numIterations - Number of iterations for the snake (e.g., 70)
% Output:
%   lesionMask - Binary mask of the segmented lesion (shrink-only)

if nargin < 2
    numIterations = 80;
end

% Convert image to grayscale
grayImg = rgb2gray(im2double(image));

% User draws rough outline around lesion (interior)
figure, imshow(image); title('Draw rough outline INSIDE lesion');
roi = drawfreehand();
wait(roi);
interiorMask = createMask(roi);

% Expand to get outer initialization (so snake shrinks)
se = strel('disk', 10);  % Expand radius (adjustable)
initMask = imdilate(interiorMask, se);

% Show initial mask
figure; imshow(initMask); title('Initial Expanded Mask (Snake Will Shrink)');

% Run Chan-Vese active contour (region-based)
lesionMask = activecontour(grayImg, initMask, numIterations, 'Chan-Vese');

% Show final result
figure; imshow(labeloverlay(image, lesionMask));
title(sprintf('Shrink-Only Chan-Vese Segmentation (Iterations = %d)', numIterations));
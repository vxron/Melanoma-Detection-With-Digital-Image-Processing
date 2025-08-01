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


% ----------------------- 2 GUI TO DRAW MASK ----------------------
% Let user manually trace the lesion area using freehand tool
disp('Draw rough ROI around the lesion and double-click to finish.');
figure('Name', 'Draw ROI', 'NumberTitle', 'off');
imshow(im); title('Draw Rough ROI Around Lesion, Then Double-Click!');
roi = drawfreehand();
wait(roi);                        % Wait for user to finish
userMask = createMask(roi);       % Converts drawing to binary mask
% 3. Use ROI to extract a bounding box from the image
props = regionprops(userMask, 'BoundingBox');
cropBox = round(props(1).BoundingBox);
% Crop image and ROI to the region
croppedImg = im(cropBox(2):(cropBox(2)+cropBox(4)-1), ...
                  cropBox(1):(cropBox(1)+cropBox(3)-1), :);
figure, imshow(croppedImg);
title('2. Initial User-Drawn Mask');

% ----------------------- 1 IMAGE PREPROCESSING --------------------

% 1.1. Median Filtering (on each RGB channel separately)
% Goal: Remove impulse noise and reduce hair streaks without affecting color structure.
med_filt_length = 5; % 5x5 window per dsp1.pdf
imMed = zeros(size(croppedImg), 'like', croppedImg);
for c = 1:size(croppedImg, 3)
    imMed(:,:,c) = medfilt2(croppedImg(:,:,c), [med_filt_length med_filt_length]);
end
figure, imshow(imMed);
title('1.1. Median Filtered RGB Image');

% 1.2. Morphological Closing (on grayscale version just for hair detection cleanup)
% Goal: Fill narrow dark gaps (e.g., hair streaks). Only needed once for cleaning.
grayForMorph = rgb2gray(imMed);
se = strel('disk', 3); % disk size empirically chosen from dsp1.pdf
imMorph = imclose(grayForMorph, se);
figure, imshow(imMorph);
title('1.2. Morphological Closing (Grayscale)');

% 1.3. Adaptive Histogram Equalization
% Goal: Improve local contrast in poorly lit images without global brightness shift.
imEq = adapthisteq(imMorph, 'ClipLimit', 0.005); % Lower clipLimit = less contrast boost;  % Apply on grayscale to guide lightness balance
figure, imshow(imEq);
title('1.3. Adaptive Histogram Equalized Image (Grayscale)');

% 1.4. Gaussian Smoothing (on full RGB image before color-based clustering)
% Goal: Remove fine noise and texture that could disrupt clustering.
sigma = 1;
g_filt_length = 4;
G = fspecial('gaussian', [g_filt_length g_filt_length], sigma);
imSmooth = zeros(size(imMed), 'like', imMed);
for c = 1:3
    imSmooth(:,:,c) = conv2(double(imMed(:,:,c)), G, 'same');
end
imSmooth = uint8(imSmooth);
figure, imshow(imSmooth);
title('1.4. Gaussian Filtered RGB Image');

% --------------------- 2 IMAGE SEGMENTATION -----------------------
% SEGMENTATION USING K-MEANS CLUSTERING IN CIELab COLOR SPACE
% Goal: Segment the lesion using color clustering in a*b* space (lighting invariant)

% 2.1. Convert to CIELab
% Goal: Convert the preprocessed RGB image (imSmooth) to L*a*b* color space
labIm = rgb2lab(im2double(imSmooth));
figure, imshow(labIm(:,:,1), []); title('2.1. L* Channel (Lightness)');
figure, imshow(labIm(:,:,2), []); title('2.1. a* Channel (Red-Green)');
figure, imshow(labIm(:,:,3), []); title('2.1. b* Channel (Blue-Yellow)');

% Region growing segmentation using L* channel
L_tolerance = 10;  % can tweak depending on how inclusive you want it; increasing for some images rlly helps
roiLocal = userMask(cropBox(2):(cropBox(2)+cropBox(4)-1), ...
                    cropBox(1):(cropBox(1)+cropBox(3)-1));

lesionMask = regionGrowLab(labIm, roiLocal, L_tolerance);

% Show result
figure, imshow(lesionMask); title('2.1. Region-Grown Lesion Mask');

% Fill the region and smooth the boundary
%lesionMask = imfill(lesionMask, 'holes');                   % Fill enclosed holes

lesionMask = bwconvhull(lesionMask, 'objects');  % Encloses all regions fully


lesionMask = imclose(lesionMask, strel('disk', 3));         % Bridge small gaps
lesionMask = imopen(lesionMask, strel('disk', 1));          % Smooth the edge
% Remove small speckle noise (e.g., tiny blobs less than 30 pixels)
minBlobArea = 10;  % adjust based on your image resolution and lesion size
lesionMask = bwareaopen(lesionMask, minBlobArea);
figure, imshow(lesionMask); title('2.1. Cleaned Region-Grown Lesion Mask');

% 2.2. Apply mask to RGB
% Reconstruct lesionMask to full-size image
fullMask = false(size(im,1), size(im,2));
fullMask(cropBox(2):(cropBox(2)+cropBox(4)-1), ...
         cropBox(1):(cropBox(1)+cropBox(3)-1)) = lesionMask;

maskedRgbImage = bsxfun(@times, im, cast(fullMask, 'like', im));
figure, imshow(maskedRgbImage); title('2.2. RGB Lesion Isolated via Region Growing');









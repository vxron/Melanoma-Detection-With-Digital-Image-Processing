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
sigma = 1.2;
g_filt_length = 9;
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

% 2.2. K-means Clustering (according to a*b*) using Custom Lab Function
% Segment & Reconstruct cluster labels into image form
pixel_labels = segmentKmeansAB(labIm, 3, 20, 10);
figure, imshow(label2rgb(pixel_labels)), title('2.2. K-means Cluster Labels (RGB-coded)');

% 2.3. Choose Lesion Cluster (Compare Two Methods)
% --- Method A: L* Intensity (lowest brightness is likely the lesion)
L = labIm(:,:,1);  % Lightness channel
nColors = 3;
meanL = zeros(nColors, 1);
for k = 1:nColors
    meanL(k) = mean(L(pixel_labels == k));
end
[~, lesion_Lstar] = min(meanL);  % Darkest cluster

% most important heuristic --> cluster should be "circular" in shape (not
% hella long)
% acc most important heuristic --> cluster should be the one that has all
% points hella close together 

% 2.4. Decide Best Cluster
% Combine both heuristics: choose the cluster that appears in both top 2 for darkness & area

% (Debugging) Show the two options and the final chosen one
figure, imshow(pixel_labels == lesion_Lstar); title('2.3A. Lesion Candidate by L* Intensity');

% 2.5. Create Binary Mask
Ibin = pixel_labels == lesion_Lstar;
figure, imshow(Ibin); title('2.5. Initial Binary Mask (Pre-cleanup)');

% 2.6. Clean Up Mask
% Fill holes, keep only l
Ibin = imfill(Ibin, 'holes');
Ibin = bwareafilt(Ibin, 1);                      % keep largest region
Ibin = imopen(Ibin, strel('disk', 2));           % smooth border
figure, imshow(Ibin); title('2.6. Cleaned Binary Mask');

% 2.7. Apply Mask to RGB Image
% Isolate lesion in RGB space
maskedRgbImage = bsxfun(@times, croppedImg, cast(Ibin, 'like', croppedImg));
figure, imshow(maskedRgbImage); title('2.7. RGB Lesion Isolated via Mask');






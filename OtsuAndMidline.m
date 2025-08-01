clc; close all; clear;

% ------------------------ 0. IMAGE LOADING ------------------------
[FILENAME, PATHNAME] = uigetfile('*.jpg', 'Select a Skin Lesion Image');
filePath = strcat(PATHNAME, FILENAME);
fprintf('Selected image: %s\n', filePath);
I = imread(filePath);
figure, imshow(I); title('Original Image');

% ------------------------ 1. PREPROCESSING ------------------------
Igray = rgb2gray(I);

% 1.1 Median Filtering
Igray = medfilt2(Igray, [5 5]);

% 1.2 Morphological Closing (removes hair streaks)
se = strel('disk', 3);
Iclosed = imclose(Igray, se);

% 1.3 Adaptive Histogram Equalization (contrast)
Ienhanced = adapthisteq(Igray, 'ClipLimit', 0.005, 'NumTiles', [4 4]); % Lower clipLimit = less contrast boost
figure, imshow(Ienhanced); title('Preprocessed Grayscale Image');

% ------------------------ 2. OTSU SEGMENTATION ------------------------
otsuLevel = graythresh(Ienhanced);
otsuMask = imbinarize(Ienhanced, otsuLevel);
otsuMask = imcomplement(otsuMask);
otsuMask = bwareafilt(otsuMask, 1); % keep largest object

% Analyze Otsu region
stats = regionprops(otsuMask, 'Area', 'Centroid', 'BoundingBox');
[maxArea, idx] = max([stats.Area]);
[rows, columns]=size(Igray);
centroid = stats(idx).Centroid;  % This is [x, y]
xCentroid = centroid(1);         % Extract scalar x
yCentroid = centroid(2);         % Extract scalar y
imgCenter = [size(Igray,2)/2, size(Igray,1)/2];
centerDist = sqrt((xCentroid - imgCenter(1))^2 + (yCentroid - imgCenter(2))^2);
imageDiag = sqrt(size(Igray,1)^2 + size(Igray,2)^2);
disp("imageDiag = "); disp(imageDiag);
whos imageDiag
isCentered = centerDist < 0.25 * imageDiag;
disp("isCentered = "); disp(isCentered);
whos isCentered
otsuArea = stats(idx).Area;
areaThreshold = 0.01 * numel(Igray);  % 1% of image area
% Optional: reject if bounding box touches image edge
bbox = stats(idx).BoundingBox;
touchesEdge = (bbox(1) < 5) || (bbox(2) < 5) || ...
              (bbox(1)+bbox(3) > size(Igray,2)-5) || ...
              (bbox(2)+bbox(4) > size(Igray,1)-5);

% ------------------ 3. MIDPOINT SEGMENTATION (if needed) ------------------
if ~isCentered || otsuArea < areaThreshold || touchesEdge
    fprintf('⚠️ Otsu segmentation not ideal. Switching to midpoint background subtraction.\n');
    bgEstimate = imgaussfilt(Igray, 25);
    subtracted = imsubtract(Igray, bgEstimate);
    midpointMask = imbinarize(subtracted, 'adaptive', 'ForegroundPolarity', 'dark');
    midpointMask = bwareafilt(midpointMask, 1);
    initMask = midpointMask;
else
    fprintf('✅ Otsu segmentation looks good. Using Otsu mask.\n');
    initMask = otsuMask;
end

% ------------------------ 4. ACTIVE CONTOUR REFINEMENT ------------------------
refinedMask = activecontour(Igray, initMask, 300, 'Chan-Vese');
refinedMask = imfill(refinedMask, 'holes');
refinedMask = imopen(refinedMask, strel('disk', 2));
figure, imshow(refinedMask); title('Final Refined Lesion Mask');

% ------------------------ 5. APPLY MASK TO RGB IMAGE ------------------------
maskedRGB = bsxfun(@times, I, cast(refinedMask, 'like', I));
figure, imshow(maskedRGB); title('Masked RGB Lesion');

% ------------------------ 6. FEATURE EXTRACTION ------------------------
stats = regionprops(refinedMask, 'Centroid', 'Orientation', 'Perimeter', 'Area', 'BoundingBox');

% A - Asymmetry (flip + overlap method)
%Obtain Centrioid Values for Blobs
Centroids=[stats.Centroid];
%Obtain Centroid Values of Biggest Blob
xCentroid = Centroids(2*(idx-1)+1);
yCentroid = Centroids(2*(idx-1)+2);
%Obtain Values to Shift Largest Blob to the Middle
middlex = columns/2;
middley = rows/2;
deltax = middlex - xCentroid;
deltay = middley - yCentroid;
%Move Blob to the Middle
binaryImage = imtranslate(refinedMask,  [deltax, deltay]);
%Rotatation of Binary Image
Orientations=[stats.Orientation];
angle = -1.*Orientations(idx);
rotatedImage = imrotate(binaryImage, angle, 'crop');
%Find Image Properties of Rotated Binary Image
props2=regionprops(rotatedImage,'all');
%Flip and Overlap Rotated Image With Itself
flipped = flipud(rotatedImage);
overlapped = flipped & rotatedImage;
%Find Area That Doesn't Overlap
nonOverlapped = xor(flipped, rotatedImage);
numberOfNonZeros = nnz(nonOverlapped);
%Asymmetry Index Calculation
AI = (numberOfNonZeros / maxArea) * 100;

% B - Border irregularity (circularity)
P = stats.Perimeter;
A = stats.Area;
circularity = (P^2) / (4 * pi * A);  % >1.5 usually considered irregular

% C - Color variation (mean ΔE in Lab)
lab = rgb2lab(im2double(I));
L = lab(:,:,1); Achannel = lab(:,:,2); Bchannel = lab(:,:,3);
Lmean = mean(L(refinedMask)); Amean = mean(Achannel(refinedMask)); Bmean = mean(Bchannel(refinedMask));
deltaE = sqrt((L - Lmean).^2 + (Achannel - Amean).^2 + (Bchannel - Bmean).^2);
colorVar = mean(deltaE(refinedMask));

% ------------------------ 7. DIAGNOSIS ------------------------
% TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*D (classic rule of thumb)
TDS = 1.3 * (AI/100) + 0.1 * circularity + 0.5 * (colorVar / 10); % normalized

if TDS < 1.5
    result = 'Benign (TDS < 4.75)';
elseif TDS <= 2.5
    result = 'Suspicious (TDS 4.8–5.45)';
else
    result = 'Malignant (TDS > 5.45)';
end

% ------------------------ 8. DISPLAY RESULT ------------------------
fprintf('\n=== Feature Results ===\n');
fprintf('Asymmetry Index (A): %.2f%%\n', AI);
fprintf('Circularity (B): %.2f\n', circularity);
fprintf('Color Variation ΔE (C): %.2f\n', colorVar);
fprintf('Total Dermoscopy Score (TDS): %.2f\n', TDS);
fprintf('>>> Final Diagnosis: %s\n', result);

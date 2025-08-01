clc;
close all;

% ----------------------- 0. IMAGE SELECTION -----------------------
[FILENAME, PATHNAME] = uigetfile('*.jpg', 'Select the Lesion Image');
filePath = strcat(PATHNAME, FILENAME);
disp('Selected image:'); disp(filePath);

% Load and display
im = imread(filePath);
figure, imshow(im); title('0. Original Lesion Image');

% ----------------------- 1. SELECT SEED POINT ---------------------
disp('Click a point near the center of the lesion (not on glare).');
figure, imshow(im); title('1. Select Seed Point Near Lesion Center');
[x, y] = ginput(1);  % Get one click
seed = round([y, x]);  % MATLAB is (row, col) = (y, x)

% ----------------------- 2. PREPROCESSING -------------------------

% --- Median filter each channel ---
imMed = zeros(size(im), 'like', im);
for c = 1:3
    imMed(:,:,c) = medfilt2(im(:,:,c), [5 5]);
end

% --- Optional Gaussian Smoothing ---
sigma = 1.2;
G = fspecial('gaussian', [9 9], sigma);
imSmooth = zeros(size(imMed), 'like', imMed);
for c = 1:3
    imSmooth(:,:,c) = conv2(double(imMed(:,:,c)), G, 'same');
end
imSmooth = uint8(imSmooth);

% --- Convert to Lab ---
labIm = rgb2lab(im2double(imSmooth));

% --- CLAHE on L* only ---
L = labIm(:,:,1);
L_eq = adapthisteq(L/100, 'ClipLimit', 0.005) * 100;  % Normalize to [0,1] then scale back
labIm(:,:,1) = L_eq;

% (Optional) --- Morph closing on L* ---
se = strel('disk', 3);
labIm(:,:,1) = imclose(labIm(:,:,1), se);

% ------------------------ 4. REGION GROWING ------------------------
% You can tune tolerance to 6–12 based on edge softness
L_tolerance = 8;
lesionMask = regionGrowLab(labIm, seed, L_tolerance);
% ------------- Postprocessing -------------
lesionMask = imfill(lesionMask, 'holes');
lesionMask = bwareafilt(lesionMask, 1);
lesionMask = imopen(lesionMask, strel('disk', 2));

% ------------------------ 5. FINAL RESULTS -------------------------
% Apply mask to RGB image
maskedRgb = bsxfun(@times, im, cast(lesionMask, 'like', im));

figure, imshow(lesionMask); title('5.1. Final Binary Lesion Mask');
figure, imshow(maskedRgb); title('5.2. Segmented Lesion in RGB');


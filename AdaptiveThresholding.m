% Otsu's Method with Adaptive Thresholding

clc;
close all;

% STYLING
% Camel Case for Variable Names
% Underscored for Constant Parameters

% ----------------------- 0 IMAGE READING GUI ----------------------
[FILENAME, PATHNAME] = uigetfile('*.jpg', 'Select the Image');
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
% 1.1. Gray Scale
iGray = rgb2gray(croppedImg);
figure,imshow(uint8(iGray));
title('1.1. Grayscaled Image:');

% 1.2. Median Filtering
% Goal: Remove impulse noise, small artifacts, salt-and-pepper pixels
med_filt_length = 5; % 5x5 from dsp1.pdf; larger window removes more noise and wider hairs but risks over-smoothing smaller lesion details
iMed = medfilt2(iGray, [med_filt_length med_filt_length]); 
figure,imshow(uint8(iMed));
title('1.2. Med Filtered Image:');

% 1.3. Morphological Closing
% Goal: Remove dark, crack-like artifacts (i.e. hair) by closing gaps in the foreground
% Implementation: Disk moves over the image and applies dilation (expand white/bright regions) -->
% erosion (resets white regions to normal size, but now holes/cracks are gone)
disk_radius = 3;
se = strel('disk', 3); % dsp1.pdf
iClosed = imclose(iMed, se);
figure,imshow(uint8(iClosed));
title('1.3. Morphologically Closed Image:');

% POSSIBLE THAT THIS SHOULD ONLY BE DONE IN PART, i.e. not FULL
% equalization
% 1.4.Contrast Enhancement (Histogram Equalization)
% Goal: Improve lesion-skin contrast by equalizing pixel intensities, especially in poorly lit images
% iEq = histeq(iClosed);
iEq = iClosed;
figure,imshow(uint8(iEq));
title('1.4. Histogram Equalized Image:');

%%% IDK IF WE REALLY NEED THIS YET
% 1.5. Gaussian Smoothing Filter (ref. dsp7.pdf)
% Goal: Further smooth out high frequency noise and small texture variations before segmentation (at the cost of blur)
sigma = 0.6; % Variance; shouldn't be too big since we don't want to cause significant blurring
g_filt_length = 4; % rule of thumb: filtersize = 6*sigma, rounded to next odd number
G = fspecial('gaussian', [g_filt_length g_filt_length], sigma);
iSmooth = conv2(double(iEq), G, 'same'); % Convolution with Gaussian filter
figure, imshow(uint8(iSmooth));
title('1.5. Gaussian Filter Output:');
iSmooth = iEq;


% --------------------- 2 IMAGE SEGMENTATION -----------------------
% Otsu's Segmentation Method (another paper.pdf and dsp1.pdf)
% Goal: Segment the lesion (foreground) from surrounding healthy skin
% (background) by finding a global intensity threshold that best separates
% two classes: lesion pixels & skin/background pixels
% Size of the CROPPED image (iSmooth)
[croppedRows, croppedCols] = size(iSmooth);
lowerBound = min(croppedRows,croppedCols)*0.5; % not too big for smaller dimension
upperBound = max(croppedRows,croppedCols)*0.07; % about 10% of max dimension
blockSize = round(min(upperBound,lowerBound));
disp(blockSize)
overlap = round(blockSize / 1.3); % ~75% overlap
step = blockSize - overlap;

disp(croppedCols);
disp(croppedRows);

% Initialize accumulation maps
iAdaptiveSum = zeros(size(iSmooth));
weightMap = zeros(size(iSmooth));

% Only iterate over pixels within the cropped region (ROI)
maxRowStart = croppedRows - blockSize + 1;
maxColStart = croppedCols - blockSize + 1;

for r = 1:step:maxRowStart
    for c = 1:step:maxColStart
        % Define block bounds safely
        r_end = r + blockSize - 1;
        c_end = c + blockSize - 1;
        block = iSmooth(r:r_end, c:c_end);

        % Normalize and compute threshold
        blockNorm = uint8(mat2gray(block)*255);
        thresh = graythresh(blockNorm);
        bwBlock = imbinarize(blockNorm, thresh);
        bwBlock = imcomplement(bwBlock);  % Assume lesion is darker

        % Add to accumulation maps
        iAdaptiveSum(r:r_end, c:c_end) = iAdaptiveSum(r:r_end, c:c_end) + double(bwBlock);
        weightMap(r:r_end, c:c_end) = weightMap(r:r_end, c:c_end) + 1;
    end
end

% Avoid divide-by-zero
weightMap(weightMap == 0) = 1;
iAdaptiveAvg = iAdaptiveSum ./ weightMap;
iAdaptive = iAdaptiveAvg > 0.45;
figure;
imagesc(iAdaptiveAvg); colormap('hot'); colorbar;
title('Adaptive Otsu Probability Map');

%{
labImg = rgb2lab(im2double(croppedImg));  % reuse this if you already did LAB
L = labImg(:,:,1);  % L* from 0 (black) to 100 (white)

% Threshold: keep pixels darker than X
% Choose neighborhood size (tune as needed)
windowSize = round(blockSize * 1.2);  % pixels (adjust for image size!)
G = fspecial('gaussian', windowSize, windowSize / 6);
localLMean = imfilter(L, G, 'replicate');
L_std = std(L(:));  % standard deviation of L* channel
offset = max(3, round(L_std / 4));  % dynamic offset

darkLocalMask = L < (localLMean - offset);
figure('Name', 'Debug: Relative Darkness Mask', 'NumberTitle', 'off');
subplot(1,3,1);
imshow(L, []); title('L* Channel (Raw Lightness)');
colormap(gca, 'gray'); colorbar;

subplot(1,3,2);
imshow(localLMean, []); title(sprintf('Local Mean L* (Window = %d)', windowSize));
colormap(gca, 'gray'); colorbar;

subplot(1,3,3);
imshow(darkLocalMask); title(sprintf('darkLocalMask (offset = %d)', offset));

iDarkFiltered = iAdaptive & darkLocalMask;
%}

% Clean and fill mask
iFilled = imfill(iAdaptive, 'holes');
iCleaned = bwareafilt(iFilled, 1);  % now keep largest clean blob

% --------------------- Final: Reconstruct Full-Size Mask & Overlay ---------------------
fullMask = false(size(im,1), size(im,2));
fullMask(cropBox(2):(cropBox(2)+cropBox(4)-1), ...
         cropBox(1):(cropBox(1)+cropBox(3)-1)) = iCleaned;

% Overlay
maskedRgb = bsxfun(@times, im, cast(fullMask, 'like', im));

% Show result
figure, imshow(fullMask); title('2.2 Final Full-Size Mask');
figure, imshow(maskedRgb); title('2.2 Final Masked RGB Image');


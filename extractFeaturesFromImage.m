function features = extractFeaturesFromImage(imagePath)
% Extracts 4 features from a lesion image (no user interaction)
% Output: [AsymmetryIndex, SuspiciousColorCount, Circularity, DarknessScore]

    % Read input image
    rgbImg = imread(imagePath);

    % --- Limit size to max 500 x 700 (preserve aspect ratio) ---
    maxH = 500;
    maxW = 700;
    [H, W, ~] = size(rgbImg);
    if H > maxH || W > maxW
        scale = min(maxH / H, maxW / W);
        newH = round(H * scale);
        newW = round(W * scale);
        rgbImg = imresize(rgbImg, [newH, newW]); % bilinear for image
    end

    % --- Preprocessing ---
    grayImg = preprocessGrayscale(rgbImg);
    colorImg = preprocessColor(rgbImg);

    % --- Segmentation (Otsu) ---
    mask = segmentOtsu(grayImg);
    %labImg = rgb2lab(im2double(colorImg));
    %mask=segmentKmeansAB2(labImg, 3, 50, 8);

    % --- Feature Extraction ---
    % Convert to full-size mask (no cropping here)
    lesionMask = mask;

    % RGB-masked image
    rgbMasked = bsxfun(@times, rgbImg, cast(lesionMask, 'like', rgbImg));

    AI = computeAsymmetryIndex(lesionMask);
    circularity = computeCircularity(lesionMask);
    darkness = computeDarknessRelativeToBackground(rgbMasked, rgbImg);
    suspiciousColors = countSuspiciousColors(rgbMasked);
    edgePixels = edge(lesionMask, 'sobel');
    borderAbrupt = sum(edgePixels(:)) / nnz(lesionMask);

    features = [AI, suspiciousColors, circularity, darkness, borderAbrupt];
end

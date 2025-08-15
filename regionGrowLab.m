function lesionMask = regionGrowLab(labImg, seedOrMask, L_tol, isDebug)
% regionGrowLab - Region growing in L*a*b* space from seed or user ROI mask.

% ------------- Setup -------------
max_bad_in_row = 2;  % how many consecutive "too bright" pixels allowed
% Extract pixel channels
L = labImg(:,:,1);
a = labImg(:,:,2);
b = labImg(:,:,3);
[rows, cols] = size(L);
visited = false(rows, cols);
lesionMask = false(rows, cols);
bad_counts = zeros(rows, cols);
% Global tolerance multiplier for seed comparison
tol_seed = L_tol * 1.8;

distanceMap_region = nan(rows, cols);
distanceMap_seed   = nan(rows, cols);

% ------------- Get Seed -------------
if islogical(seedOrMask)
    % Find darkest pixel inside the mask (lowest L* value)
    L = labImg(:,:,1);
    L_masked = L;
    L_masked(~seedOrMask) = NaN;

    [minVal, linearIdx] = min(L_masked(:));
    [rIdx, cIdx] = ind2sub(size(L_masked), linearIdx);

    seed = [cIdx, rIdx];
elseif isnumeric(seedOrMask) && numel(seedOrMask) == 2
    if isDebug
        disp("detected seed!")
    end
    seed = round(seedOrMask);
else
    error('Second argument must be either a binary mask or [x, y] seed point.');
end

% ======= BEGIN ADAPTIVE FTR WEIGHT CALCULATION =======

% Define initial lesion mask if not given
if islogical(seedOrMask)
    lesionMaskInit = seedOrMask;
else
    lesionMaskInit = false(size(L));
    lesionMaskInit(seed(2), seed(1)) = true;
end

% Mean L* inside lesion
meanL_lesion = mean(L(lesionMaskInit));

% Approximate background skin by dilating lesion and subtracting lesion
se = strel('disk', 5); % 5-pixel dilation
ringMask = imdilate(lesionMaskInit, se) & ~lesionMaskInit;
meanL_skin = mean(L(ringMask));

% Contrast between lesion and skin
contrastL = abs(meanL_skin - meanL_lesion);

% Color variation inside lesion in a* and b*
std_a = std(a(lesionMaskInit));
std_b = std(b(lesionMaskInit));

disp(meanL_lesion)
disp(contrastL)
% Set weights adaptively
if contrastL > 10 && meanL_lesion < 70
    w_L = 8.7; % High contrast & dark lesion so we emphasize L*
else
    w_L = 6.5; % Otherwise lower weight
end

if std_a > 5
    w_a = 2.5; % lots of variation within lesion means we dont want to use this as segmentation criteria
else
    w_a = 1.0;
end

if std_b > 5
    w_b = 2.5;
else
    w_b = 1.0;
end

if contrastL < 10
    w_x = 0.1; % Low contrast so we rely more on spatial continuity
    w_y = 0.1;
else
    w_x = 0.5;
    w_y = 0.5;
end

% Convert seed to row, col
seedRow = seed(2); seedCol = seed(1);

if isDebug
    figure, imshow(L, []); hold on;
    plot(seedCol, seedRow, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
    title('Seed Point for Region Growing (L* channel)');
end


% ------------- Region Growing -------------
% Extract seed feature vector
seedFeature = [L(seedRow, seedCol), a(seedRow, seedCol), b(seedRow, seedCol), seedCol, seedRow];
regionSum = seedFeature;
regionCount = 1;
queue = [seedRow, seedCol];  % Initialize queue with the starting seed pixel (row, col) and parent pixel
% now each queue element has struct [row col L a b x y]
% Define 8-connected neighbors
neighbors = [ -1  0;
               1  0;
               0 -1;
               0  1;
              -1 -1;
              -1  1;
               1 -1;
               1  1];

% Loop until all connected pixels that meet the criteria have been visited
while ~isempty(queue)
    pt = queue(1, :);
    queue(1,:) = []; % pop first (FIFO)

    r = pt(1); c = pt(2);

    % Skip this pixel if it's outside the image bounds or already visited
    if r < 1 || r > rows || c < 1 || c > cols || visited(r,c)
        continue;
    end

    visited(r,c) = true;

    % Current pixel's 5D feature
    currFeature = [L(r,c), a(r,c), b(r,c), c, r];
    regionMean = regionSum / regionCount;

    % Distance to parent region
    diffP = currFeature - regionMean;
    d_parent = sqrt((w_L*diffP(1))^2 + (w_a*diffP(2))^2 + (w_b*diffP(3))^2 + (w_x*diffP(4))^2 + (w_y*diffP(5))^2);

    % Distance to seed
    diffS = currFeature - seedFeature;
    d_seed = sqrt((w_L*diffS(1))^2 + (w_a*diffS(2))^2 + (w_b*diffS(3))^2 + (w_x*diffS(4))^2 + (w_y*diffS(5))^2);

    % Store distances
    distanceMap_region(r, c) = d_parent;
    distanceMap_seed(r, c) = d_seed;

    % Accept pixel if within both local and global tolerances
    if d_parent <= L_tol && d_seed <= tol_seed
        lesionMask(r,c) = true;
        regionSum = regionSum + currFeature;
        regionCount = regionCount + 1;
        bad_counts(r,c) = 0;
        for k = 1:size(neighbors,1)
            nr = r + neighbors(k,1);
            nc = c + neighbors(k,2);
            if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols && ~visited(nr,nc)
                queue(end+1,:) = [nr, nc]; %#ok<AGROW>
            end
        end
    else
        bad_counts(r,c) = bad_counts(r,c) + 1;
        if bad_counts(r,c) <= max_bad_in_row
            for k = 1:size(neighbors,1)
                nr = r + neighbors(k,1);
                nc = c + neighbors(k,2);
                if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols && ~visited(nr,nc)
                    queue(end+1,:) = [nr, nc]; %#ok<AGROW>
                end
            end
        end
    end

end

if isDebug
    figure;
    subplot(1,2,1); imagesc(distanceMap_region); colorbar; title('Distance to Parent');
    subplot(1,2,2); imagesc(distanceMap_seed); colorbar; title('Distance to Seed');
    
    figure, imshow(L, []); hold on;
    plot(seed(1), seed(2), 'r+', 'MarkerSize', 14, 'LineWidth', 2);
    title('Seed Point on L* Channel');
    
    figure; imshow(labeloverlay(L, lesionMask));
    title('Region Growing: Region + Seed Distance');
end
end

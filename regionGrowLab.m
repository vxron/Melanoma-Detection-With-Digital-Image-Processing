function lesionMask = regionGrowLab(labImg, seedOrMask, L_tol)
% regionGrowLab - Region growing in L*a*b* space from seed or user ROI mask.
%
% Inputs:
%   labImg     - CIELab image (use rgb2lab(im2double(...)))
%   seedOrMask - Either a binary ROI mask OR a seed coordinate [x, y]
%   L_tol      - Tolerance threshold for L* similarity (e.g., 6–12)
%
% Output:
%   lesionMask - Binary mask for lesion

% ------------- Setup -------------
max_bad_in_row = 3;  % how many consecutive "too bright" pixels allowed
% Extract pixel channels
L = labImg(:,:,1);
a = labImg(:,:,2);
b = labImg(:,:,3);
[rows, cols] = size(L);
visited = false(rows, cols);
lesionMask = false(rows, cols);
bad_counts = zeros(rows, cols);
% Global tolerance multiplier for seed comparison
tol_seed = L_tol * 1.25;

distanceMap_region = nan(rows, cols);
distanceMap_seed   = nan(rows, cols);


% ---------------------- Feature Weights --------------------------
w_L = 5.0;     % Strong weight for L*
w_a = 0.6;     % Moderate for a*
w_b = 0.6;     % Moderate for b*
w_x = 0.05;     % Low for X spatial
w_y = 0.05;     % Low for Y spatial

% ------------- Get Seed -------------
if islogical(seedOrMask)
    % It's a mask → use centroid as seed
    props = regionprops(seedOrMask, 'Centroid');
    seed = round(props(1).Centroid);  % [x, y]
elseif isnumeric(seedOrMask) && numel(seedOrMask) == 2
    disp("detected seed!")
    seed = round(seedOrMask);  % assume [x, y] directly
else
    error('Second argument must be either a binary mask or [x, y] seed point.');
end

% Convert seed to row, col
seedRow = seed(2); seedCol = seed(1);

% Debug: Show seed on image
figure, imshow(L, []); hold on;
plot(seedCol, seedRow, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
title('Seed Point for Region Growing (L* channel)');

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

% Visualize distances
figure;
subplot(1,2,1); imagesc(distanceMap_region); colorbar; title('Distance to Parent');
subplot(1,2,2); imagesc(distanceMap_seed); colorbar; title('Distance to Seed');



figure, imshow(L, []); hold on;
plot(seed(1), seed(2), 'r+', 'MarkerSize', 14, 'LineWidth', 2);
title('Seed Point on L* Channel');

figure; imshow(labeloverlay(L, lesionMask));
title('Region Growing: Region + Seed Distance');
end

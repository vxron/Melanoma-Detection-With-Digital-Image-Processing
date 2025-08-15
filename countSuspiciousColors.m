function nColors = countSuspiciousColors(rgbMask)
    lesionPixels = reshape(rgbMask, [], 3);
    lesionPixels = lesionPixels(any(lesionPixels > 0, 2), :);
    lesionLab = rgb2lab(double(lesionPixels)/255);
    K = 7;
    [idx, centroids] = kmeans(lesionLab, K, 'MaxIter', 200, 'Start', 'plus', 'Replicates', 2, 'EmptyAction', 'singleton');

    suspiciousColors = struct();
    suspiciousColors.Black = [0.06 0.27 0.10; 39.91 30.23 22.10];
    suspiciousColors.DarkBrown = [14.32 6.85 6.96; 47.57 27.14 46.81];
    suspiciousColors.LightBrown = [47.94 11.89 19.86; 71.65 44.81 64.78];
    suspiciousColors.White = [100 0 0; 100 0 0];
    suspiciousColors.Red = [54.29 80.81 69.89; 54.29 80.81 69.89];
    suspiciousColors.BlueGray = [50.28 -30.14 -11.96; 50.28 -30.14 -11.96];

    p = 3;
    minkowskiDist = @(x,y) (sum(abs(x - y).^p, 2)).^(1/p);
    T = minkowskiDist(suspiciousColors.White(1,:), suspiciousColors.Black(1,:)) / 2;
    nColors = 0;
    totalPixels = numel(idx);

    fields = fieldnames(suspiciousColors);
    for i = 1:length(fields)
        colorName = fields{i};
        if strcmp(colorName, 'Black')
            continue;
        end
        colorMin = suspiciousColors.(colorName)(1,:);
        colorMax = suspiciousColors.(colorName)(2,:);
        for cIdx = 1:size(centroids,1)
            dist = minkowskiDist(centroids(cIdx,:), colorMin);
            clusterPixels = sum(idx == cIdx);
            pixelRatio = clusterPixels / totalPixels;
            if any(strcmp(colorName, {'White', 'Red', 'BlueGray'}))
                if dist < T && pixelRatio >= 0.05
                    nColors = nColors + 1;
                    break
                end
            else
                if all(centroids(cIdx,:) >= colorMin) && all(centroids(cIdx,:) <= colorMax) ...
                        && pixelRatio >= 0.05
                    nColors = nColors + 1;
                    break
                end
            end
        end
    end
end

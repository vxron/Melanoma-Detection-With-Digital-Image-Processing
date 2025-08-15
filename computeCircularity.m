function circularity = computeCircularity(mask)
    stats = regionprops(mask, 'Area', 'Perimeter');
    if isempty(stats)
        circularity = NaN;
        return;
    end
    area = stats(1).Area;
    perimeter = stats(1).Perimeter;
    if area == 0
        circularity = NaN;
    else
        circularity = (perimeter^2) / (4 * pi * area);
    end
end

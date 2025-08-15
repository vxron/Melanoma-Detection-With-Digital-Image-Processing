function AI = computeAsymmetryIndex(mask)
    stats = regionprops(mask, 'BoundingBox', 'Area');
    if isempty(stats) || stats(1).Area == 0
        AI = 0;
        return;
    end
    bbox = round(stats(1).BoundingBox);
    subMask = mask(bbox(2):bbox(2)+bbox(4)-1, bbox(1):bbox(1)+bbox(3)-1);
    statsSub = regionprops(subMask, 'Centroid', 'Orientation', 'Area');
    if isempty(statsSub)
        AI = 0;
        return;
    end
    [~, idxMax] = max([statsSub.Area]);
    if statsSub(idxMax).Area == 0
        AI = 0;
        return;
    end
    [rows, cols] = size(subMask);
    dx = cols/2 - statsSub(idxMax).Centroid(1);
    dy = rows/2 - statsSub(idxMax).Centroid(2);
    centered = imtranslate(subMask, [dx, dy]);
    angle = -statsSub(idxMax).Orientation;
    aligned = imrotate(centered, angle, 'crop');
    flippedH = fliplr(aligned);
    flippedV = flipud(aligned);
    xorH = xor(flippedH, aligned);
    xorV = xor(flippedV, aligned);
    AI_H = nnz(xorH) / statsSub(idxMax).Area;
    AI_V = nnz(xorV) / statsSub(idxMax).Area;
    AI = 100 * (AI_H + AI_V) / 2;
end

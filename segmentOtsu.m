function mask = segmentOtsu(grayImg)
    normImg = uint8(mat2gray(grayImg)*255);
    level = graythresh(normImg);
    bw = imbinarize(normImg, level);
    bw = imcomplement(bw);
    bw = logical(bw);

    % 0) Keep only the main lesion
    bw = bwareafilt(bw, 1);
    
    % 1) Seal narrow background canals by WIDTH (no shape blobbing)
    %    Any background pixels within t px of the lesion are flipped to foreground.
    t = 3;  % seals canals up to ~2*t px wide (tune: 2..5)
    thin_bg = (~bw) & (bwdist(bw) <= t);
    bw_sealed = bw | thin_bg;
    
    % 2) Now enclosed voids are true "holes" -> fill them
    bw_filled = imfill(bw_sealed, 'holes');
    
    % 3) Optional gentle smoothing (don’t reopen holes)
    bw_smooth = imclose(bw_filled, strel('disk', 1));   % very mild
    bw_smooth = bwareaopen(bw_smooth, 20);              % remove tiny specks
    mask = bw_smooth;
    imshow(mask);
end

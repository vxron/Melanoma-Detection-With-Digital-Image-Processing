function mask = segmentOtsu(grayImg)
    normImg = uint8(mat2gray(grayImg)*255);
    level = graythresh(normImg);
    bw = imbinarize(normImg, level);
    bw = imcomplement(bw);
    bw = logical(bw);
    bw = bwareafilt(bw, 1);
    thin_bg = (~bw) & (bwdist(bw) <= t);
    bw_sealed = bw | thin_bg;
    bw_filled = imfill(bw_sealed, 'holes');
    bw_smooth = imclose(bw_filled, strel('disk', 1));
    bw_smooth = bwareaopen(bw_smooth, 20);
    mask = bw_smooth;
    imshow(mask);
end

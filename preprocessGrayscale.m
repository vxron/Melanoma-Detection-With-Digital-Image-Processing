function imgOut = preprocessGrayscale(img)
    imgGray = rgb2gray(img);
    imgMed = medfilt2(imgGray, [5 5]);
    imgClosed = imclose(imgMed, strel('disk', 3));
    sigma = 1.2;
    g_filt_length = 9; 
    G = fspecial('gaussian', [g_filt_length g_filt_length], sigma);
    imgSmooth = conv2(double(imgClosed), G, 'same');
    imgOut = uint8(imgSmooth);
end

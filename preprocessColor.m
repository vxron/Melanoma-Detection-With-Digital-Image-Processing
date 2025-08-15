function imgOut = preprocessColor(img)
    imgMed = zeros(size(img), 'like', img);
    for c = 1:3
        imgMed(:,:,c) = medfilt2(img(:,:,c), [5 5]);
    end
    labMed = rgb2lab(imgMed); 
    L = labMed(:,:,1);
    se = strel('disk', 3);
    L_closed = imclose(L, se);
    labEnhanced = labMed;
    labEnhanced(:,:,1) = L_closed;
    rgbEnhanced = lab2rgb(labEnhanced);
    rgbEnhanced = im2uint8(rgbEnhanced);
    sigma = 1;
    g_filt_length = 4;
    G = fspecial('gaussian', [g_filt_length g_filt_length], sigma);
    imgSmooth = zeros(size(rgbEnhanced), 'like', rgbEnhanced);
    for c = 1:3
        imgSmooth(:,:,c) = conv2(double(rgbEnhanced(:,:,c)), G, 'same');
    end
    imgOut = uint8(imgSmooth);
end
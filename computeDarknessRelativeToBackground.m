function darknessScore = computeDarknessRelativeToBackground(rgbMasked, fullRgbImage)
    Ilab_full = rgb2lab(fullRgbImage);
    Lchannel_full = Ilab_full(:,:,1);
    lesionMask = any(rgbMasked > 0, 3);
    backgroundMask = ~lesionMask;
    lesionLValues = Lchannel_full(lesionMask);
    backgroundLValues = Lchannel_full(backgroundMask);
    meanLesionL = mean(lesionLValues);
    meanBackgroundL = mean(backgroundLValues);
    darknessScore = max(0, (meanBackgroundL - meanLesionL) / 100);
end

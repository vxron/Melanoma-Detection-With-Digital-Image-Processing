function pixel_labels = segmentKmeansAB(imRGB, k, numIterations, convergenceThresh)
% segmentKmeansAB - Performs K-means clustering in a*b* color space (Lab)
%
% Inputs:
%   imRGB - RGB image to segment
%   k - number of clusters (e.g. 3)
%   numIterations - max iterations for K-means
%   convergenceThresh - threshold on cost difference for convergence
%
% Output:
%   pixel_labels - 2D image of cluster labels for each pixel

% Read the image 
if ischar(imRGB) || isstring(imRGB)
    imRGB = imread(imRGB);
end
labImg = rgb2lab(im2double(imRGB)); % Convert to L*a*b*
aStar = labImg(:,:,2);
bStar = labImg(:,:,3);
% Create matrix where first col is a* values, then b* values; each pixel = [a*, b*]
ab = [aStar(:), bStar(:)];
[nPixels, ~] = size(ab);

% Visualize pixel distribution in a*b* space
figure;
scatter(ab(:,1), ab(:,2), 5, '.');
title('Initial Pixel Distribution in a*b* Color Space');
xlabel('a*'); ylabel('b*');
grid on;

% Initialize K means (k random cluster centers between 0 and 1)
mean1 = ab(randperm(nPixels, k), :);  % Randomly choose k pixels as centers

% MAIN ALGORITHM 
J1 = zeros(numIterations, 1); % Cost function history
for iter = 2:numIterations
    Rnk = zeros(nPixels, k); % Responsibility matrix for this iteration
    d = zeros(nPixels, k); 
    % === Assignment Step ===
    for i=1:nPixels
        % For every pixel, loop through all cluster centers
        for j=1:k
            % Get squared distance of pixels from cluster means (nPixels x k)
            d(i,j) = norm(ab(i,:) - mean1(j,:))^2;
        end
        [~, Imin] = min(d(i,:)); % Imin is k index this pixel is closest to
        Rnk(i,Imin) = 1; % Each row represents a pixel, and puts a 1 for the pixel it's closest to (Imin) --> assignment operation matrix (nPixels x k)
    end

    % === Cost calculations ===
    J1(iter) = 0;
    sumRnk = zeros(1,k); % Row vector (to tell us how many pixels have been allocated to each mean)
    for i=1:nPixels
        for j=1:k
            J1(iter)=J1(iter)+Rnk(i,j)*d(i,j); % Total cost of all pixels with its currently allocated centers
        end
        sumRnk = sumRnk + Rnk(i,:); % For updating means, how many pixels in each cluster (1 x k row vector)  
    end
   
    % === Update means for this iteration ===
    newMeans = zeros(k, 2); % New cluster centers
    for i = 1:nPixels
        for j = 1:k
            % When Rnk is 0 (no corresponding pixels attached to the cluster), skip updating this mean to avoid div by 0
            if Rnk(i,j) == 1
                newMeans(j,:) = newMeans(j,:) + ab(i,:);
            end
        end
    end
    
    for j = 1:k
        if sumRnk(j) ~= 0 % Avoid div by zero
            mean1(j,:) = newMeans(j,:)/sumRnk(j);
        end
    end
    
    % Check difference between cost of this iteration and previous iteration
    if (abs(J1(iter)-J1(iter-1))<n1)
        break; % Break if convergence
    end

    % Assign each pixel to final cluster
    labels = zeros(nPixels,1);
    for i = 1:nPixels
        [~, labels(i)] = max(Rnk(i,:));
    end
    pixel_labels = reshape(labels, size(imRGB,1), size(imRGB,2));
        
end

















function pixel_labels = segmentKmeansAB2(labImg, k, numIterations, convergenceThresh, customWeights)
% segmentKmeansAB - Performs K-means clustering in a*b* color space (Lab)
%
% Inputs:
%   labIm - Lab color space image (output of rgb2lab(im2double(...)))
%   k - number of clusters (e.g. 3)
%   numIterations - max iterations for K-means
%   convergenceThresh - threshold on cost difference for convergence
%
% Output:
%   pixel_labels - 2D image of cluster labels for each pixel

aStar = labImg(:,:,2);
bStar = labImg(:,:,3);
Lstar = labImg(:,:,1);
Lvec = Lstar(:);
ab = [aStar(:), bStar(:)];
abl = [Lvec, ab];  % Now each pixel is [L*, a*, b*]
[nRows, nCols, ~] = size(labImg);
[nPixels, ~] = size(ab);

% Create positional features (row = y, col = x)
[X, Y] = meshgrid(1:nCols, 1:nRows);
pos = [Y(:), X(:)];  % row-major positions

% Combine color and spatial features
Xdata = [abl, pos];  % Xdata = [L*, a*, b*, y, x]

% Spatial weight parameter selected by inspection
if nargin < 5
    L_weight = 6.99;     % HIGH priority: lesion often darker
    a_weight = 0.5;     % Moderate: distinguish red/green
    b_weight = 0.5;     % Moderate: distinguish blue/yellow
    spatialWeight = 1.48; % Low–medium: just enough to keep it compact
else
    L_weight = customWeights(1);
    a_weight = customWeights(2);
    b_weight = customWeights(3);
    spatialWeight = customWeights(4);
end

% Normalize each feature range to [0, 1]
Xdata(:,1) = Xdata(:,1) / 100 * L_weight;       % L* ∈ [0, 100]
Xdata(:,2) = (Xdata(:,2)+128)/255 * a_weight;   % a* ∈ [-128, 127]
Xdata(:,3) = (Xdata(:,3)+128)/255 * b_weight;   % b* ∈ [-128, 127]
Xdata(:,4) = Xdata(:,4) / max(Xdata(:,4)) * spatialWeight;
Xdata(:,5) = Xdata(:,5) / max(Xdata(:,5)) * spatialWeight;
  
% Initialize K means (k random cluster centers between 0 and 1)
mean1 = Xdata(randperm(nPixels, k), :);  % Randomly choose k pixels as centers

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
            d(i,j) = norm(Xdata(i,:) - mean1(j,:))^2;
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
    newMeans = zeros(k, 5); % New cluster centers
    for i = 1:nPixels
        for j = 1:k
            % When Rnk is 0 (no corresponding pixels attached to the cluster), skip updating this mean to avoid div by 0
            if Rnk(i,j) == 1
                newMeans(j,:) = newMeans(j,:) + Xdata(i,:);
            end
        end
    end
    
    for j = 1:k
        if sumRnk(j) ~= 0 % Avoid div by zero
            mean1(j,:) = newMeans(j,:)/sumRnk(j);
        end
    end
    
    % Check difference between cost of this iteration and previous iteration
    if (abs(J1(iter)-J1(iter-1))<convergenceThresh)
        break; % Break if convergence
    end    
end

% Assign each pixel to final cluster
[~, labels] = max(Rnk, [], 2);
pixel_labels = reshape(labels, nRows, nCols);
















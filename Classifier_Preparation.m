imgFolder = 'Lesion_Images';
imgFiles = dir(fullfile(imgFolder, '*.jpg'));

% Initialize
n = numel(imgFiles);
features = zeros(n, 5);
labels = zeros(n, 1);  % 1 = melanoma, 0 = not

% Loop
for i = 1:n
    filename = imgFiles(i).name;
    labels(i) = contains(lower(filename), 'melanoma'); % auto-label
    imagePath = fullfile(imgFolder, filename);
    features(i, :) = extractFeaturesFromImage(imagePath);
end

% Create table
T = array2table(features, 'VariableNames', ...
    {'Asymmetry', 'SuspiciousColorCount', 'Circularity', 'Darkness', 'BorderAbrupt'});
T.Label = labels;

% Save
writetable(T, 'lesion_features.csv');

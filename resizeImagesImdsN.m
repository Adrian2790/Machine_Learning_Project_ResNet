function resizeImagesImdsN(imds, pathImages, newSize, outputPath)
    % Create a folder to save resized images
    if ~exist(outputPath, 'dir')
        mkdir(outputPath);
    end

    % Get the number of images in the datastore
    numImages = numel(imds.Files);

    % Process each image in the datastore
    for i = 1:numImages
        % Read the image
        img = readimage(imds, i);

        % Resize the image
        imgResized = imresize(img, newSize);

        % Get the path to save the resized image
        [filePath, fileName, fileExt] = fileparts(imds.Files{i});
        
        % Create subfolder if not exists
        subFolder = strrep(filePath, pathImages, outputPath);
        if ~exist(subFolder, 'dir')
            mkdir(subFolder);
        end
        
        % Save the resized image
        newFilePath = fullfile(subFolder, [fileName, fileExt]);
        imwrite(imgResized, newFilePath);
    end

    % Update the image datastore to point to the new resized images
    imds = imageDatastore(outputPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

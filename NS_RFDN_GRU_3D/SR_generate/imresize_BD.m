function ImLR = imresize_BD(ImHR, scale, kernelsize, sigma)
% ImLR and ImHR are uint8 data
% downsample by Bicubic
kernel  = fspecial('gaussian',kernelsize,sigma);
blur_HR = imfilter(ImHR,kernel,'replicate');
ImLR = imresize(blur_HR, 1/scale, 'bicubic');
end
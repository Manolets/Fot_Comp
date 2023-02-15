im = imread("foto_bw.jpg");

imshow(im);

im_min = min(im(:))
im_max = max(im(:))
im_mean = mean(im(:))

im2 = double(im)/255

imshow(sin(pi * im2))

im2d = 0.1 + randn(size(im))
im2d_min = min(im2d(:))
im2d_max = max(im2d(:))

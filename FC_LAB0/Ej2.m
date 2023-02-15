faunia = imread("faunia.jpg");

figure
subplot(2, 2, 4)
imshow(faunia);
subplot(2, 2, 1);
imshow(faunia(:,:,1))
subplot(2, 2, 2);
imshow(faunia(:,:,2))
subplot(2, 2, 3);
imshow(faunia(:,:,3))
close all
figure
imshow(faunia(:,:,[2 3 1]))

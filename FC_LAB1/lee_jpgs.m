clear
dirname='./retratos/';
lista = dir([dirname 'ret*']); L = length(lista);
fprintf('Encontradas %d imagenes en el directorio\n',L);
fprintf('Leyendo:\n'),

N=288; M=192;  imags=uint8(zeros(N,M,L)); 
for k=1:L  
  org = [dirname lista(k).name]; 
  im=imread(org);  
  if (size(im,3)>1), im=rgb2gray(im); end 
  imags(:,:,k)=im;   
  fprintf('%3d ',k); if mod(k,20)==0, fprintf('\n'); end
end
imags=im2double(imags);

%%
tX = 3456; tY = 5760;
sX = 192; sY = 288;
mosaico=zeros(tX, tY);
rx=(1:sX); ry=(1:sY);
for j=1:12
    for i=1:30
        mosaico(ry, rx) = imags(:, :, i + (j-1)*12);
        rx = rx + sX;
    end
    ry = ry + sY;
    rx = (1:sX);
end
imshow(mosaico);
%%
target = imread("target.jpg");
target = im2double(target);
rx=(1:sX); ry=(1:sY);

usos = zeros(1, 360);
timeout = usos;
PENALTY = 50;
for j=1:12
    for i=1:30
        sM = target(ry, rx);
        deltas = 1:360;
        for k=1:length(deltas)
            if timeout(k) > 0 
                timeout(k) = timeout(k) - 1; deltas(k) = 1000;continue, 
            end
            deltas(k) = mean2(abs(sM - imags(:, :, k)));
        end
        [min_val, min_index] = min(deltas);
        mosaico(ry, rx) = imags(:, :, min_index);
        usos(min_index)=usos(min_index)+1;
        timeout(min_index) = PENALTY;
        rx = rx + sX;
    end
    ry = ry + sY;
    rx = (1:sX);
end
imshow(mosaico);
%plot(usos);

%% Más resolución
target = imread("target.jpg");
target = im2double(target);
sX = 192/2; sY = 288/2;
rx=(1:sX); ry=(1:sY);
%imags = imresize(imags, 1/2);
usos = zeros(1, 360);
timeout = usos;
PENALTY = 100;
for j=1:24
    for i=1:60
        sM = target(ry, rx);
        deltas = 1:360;
        for k=1:length(deltas)
            if timeout(k) > 0 
                timeout(k) = timeout(k) - 1; deltas(k) = 1000;continue, 
            end
            deltas(k) = mean2(abs(sM - imags(:, :, k)));
        end
        [min_val, min_index] = min(deltas);
        mosaico(ry, rx) = imags(:, :, min_index);
        usos(min_index)=usos(min_index)+1;
        timeout(min_index) = PENALTY;
        rx = rx + sX;
    end
    ry = ry + sY;
    rx = (1:sX);
end
imshow(mosaico);
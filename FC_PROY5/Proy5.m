%% 1
xy = [ 0 600 465 215; 0 65 680 585];
uv = [275 655 365 25; 30 285 755 340];
p_mat = get_proy(xy, uv);
dump_mat(p_mat);

% Verificar la transformación utilizando la función convierte
uv2 = convierte(xy, p_mat);
%%
% Calcular las diferencias entre las coordenadas de destino iniciales y las obtenidas por la transformación
diff = abs(uv - uv2);
disp('Diferencias:')
disp(diff)

% Calcular la coordenada de destino para el punto (x=200,y=100)
punto = [200; 100; 1];
uv_punto = p_mat * punto;
uv_punto = uv_punto(1:2) / uv_punto(3);
disp('Coordenada de destino para el punto (200,100):')
disp(uv_punto)

%% 1.2
im = imread('foto.jpg');
im = im2double(im);
iP = inv(p_mat);
im2 = warp_img(im, iP);
imshow(im2);

%% 1.3
im = imread('billboard.jpg');
imshow(im);

% Obtener las coordenadas de las esquinas del cartel en la imagen "billboard.jpg"
dest_corners = [581 1195 579 1193; 269 175 553 651];

src_corners = [1 1 size(im,2) size(im,2); 1 size(im,1) 1 size(im,1)]';
P = get_proy(dest_corners, src_corners);

im2 = warp_img(im, inv(P));

%% 3
im = imread('img.jpg');
im = im2double(im);

[height, width, channels] = size(im); % 1152x648
aspect_ratio = width / height; % 16:9
E = energia(im); 
E_10_10 = E(10, 10); % 0.1372
mean_E = mean2(E); % 0.5287

figure;
imagesc(E);
colormap('jet');
colorbar;

%%

new_width = round(height * 3 / 2);
n = size(im, 2) - new_width;

im_reduced = im;

for i = 1:n

    E = energia(im_reduced);


    [~, min_col] = min(sum(E, 1));
    im_reduced(:, min_col, :) = [];


    E_reduced = energia(im_reduced);
end


mean_E_reduced = mean2(E_reduced); % 0.5914

figure;
subplot(1, 2, 1);
imshow(im);
title('Imagen original');

subplot(1, 2, 2);
imshow(im_reduced);
title('Imagen reducida');


imwrite(im_reduced, 'imagen_reducida.jpg');

%%
clc
clear
im = imread('img.jpg');
im = im2double(im);
E = energia(im);
M = calcula_M(E);
M(1:3, end-3:end)

% ans =

%   69.7636   69.9588   70.0499       Inf
%   69.7609   69.9192   69.7876       Inf
%   69.7625   69.7442   69.7537       Inf
%%

imagesc(M);
colormap(hot);
colorbar;

%%
[ALTO, ANCHO, ch] = size(im);
s = find_seam(M);

% plot the seam on the original image
I_seam = im;
for i = 1:ALTO
    I_seam(i, s(i), :) = [0, 255, 0];
end
figure, imshow(I_seam)
%%
[x, y] = ginput(3);
% x	[-43.4929;69.0471;69.0471]	3×1	double
% y	[-141.0062;-109.2904;-109.2904]	3×1	double
%%
% Plot the first row of M
figure;
plot(M(1,:));
title('First row of M');
xlabel('Column');
ylabel('Energy');
%%
[min_energy, min_col] = min(M(1,:));
% min_energy	68.2987	1×1	double
% min_col	1028	1×1	double

%%
energy = sum(E(sub2ind([ALTO, ANCHO], (1:ALTO)', s)));
% energy	68.2987	1×1	double

%%
im = imread('img.jpg');
im = im2double(im);
E = energia(im);
M = calcula_M(E);
s = find_seam(M);
[ALTO, ANCHO, channels] = size(im);
for i = 1:ALTO
    for j = 1:size(im, 1)
        im(j, s(j):end-1, :) = im(j, s(j)+1:end, :);
    end
    im = im(:, 1:end-1, :); 


    E = energia(im);
    M = calcula_M(E);
    s = find_seam(M);
end


E = energia(im);
M_final = calcula_M(E);

%%
figure;
plot(1:size(M_final,2), M_final(1,:));
title('Primera fila de la matriz de energía acumulada final');
xlabel('Columnas');
ylabel('Energía acumulada');

%%
imshow(im)

%%
im = imread('img.jpg');
im = im2double(im);
E = energia(im);
M = calcula_M(E);
s = find_seam(M);
[ALTO, ANCHO, channels] = size(im);
num_columnas_a_agregar = 2 * ALTO - ANCHO
% num_columnas_a_agregar	144	1×1	double

%%
% Leer la imagen original
im = imread('img.jpg');
im = im2double(im);

aspect_ratio = 2;

W_desired = round(aspect_ratio * size(im, 1));
H_desired = size(im, 1);

num_columns_to_add = 2 * H_desired - W_desired;


for i = 1:num_columns_to_add
 
    E = energia(im);
    M = calcula_M(E);
    s = find_seam(M);

    for j = size(im, 1):-1:1
        im(j, s(j)+1:end+1, :) = im(j, s(j):end, :);
        im(j, s(j), :) = im(j, s(j)-1, :);
    end
end


im_resized = imresize(im, [H_desired W_desired]);
imshow(im_resized);

%%
% Cargar imagen original y número de costuras a añadir
img = imread('img.jpg');
num_seams = 144;

% Obtener dimensiones de la imagen
[alto, ancho, ~] = size(img);

% Calcular energía de la imagen original
E = energia(img);

% Inicializar matriz S
S = zeros(alto, num_seams);

% Bucle para encontrar las n costuras
for k = 1:num_seams
    % Calcular energía acumulada
    M = calcula_M(E);

    % Encontrar la costura de energía mínima
    s = find_seam(M);

    % Guardar la costura en la matriz S
    S(:, k) = s;

    % Penalizar la energía de la última costura
     if k < num_seams
        for i = 1:alto
            j = s(i);
            E(i, j) = E(i, j) * 1.5;
        end
     end
end

% Mostrar imagen original
imshow(img);

% Superponer n costuras verticales en verde
hold on
for k = 1:num_seams
    costura_k = S(:, k);
    idx = sub2ind([alto, ancho], (1:alto)', costura_k);
    plot(costura_k, 1:alto, 'g', 'LineWidth', 2);
end
hold off
%%

[N, M, ~] = size(im);
im2 = zeros(size(im,1), N*size(im,2));

idx = zeros(1,new_length);

% Calcular la cantidad de píxeles a agregar entre cada par de píxeles duplicados
num_new_pixels = diff([0 pixels_to_duplicate new_length]);

% Iterar sobre los pares de píxeles a duplicar y agregar los nuevos píxeles
for i = 1:length(pixels_to_duplicate)
    start_idx = pixels_to_duplicate(i);
    end_idx = start_idx + num_new_pixels(i) - 1;
    idx(start_idx:end_idx) = i;
end

% Llenar los valores restantes de idx
last_nonzero_idx = find(idx,1,'last');
idx(last_nonzero_idx+1:end) = length(pixels_to_duplicate)+1;

% Imprimir la fila original y el vector idx resultante
disp('Original row:');
disp(original_row);
disp('idx:');
disp(idx);
%%
% Crear una imagen im2 de ceros del tamaño necesario
[N, M, ~] = size(img);
im2 = zeros(size(img,1), N*size(img,2));

% Para cada fila k en la imagen original
for k = 1:size(img,1)
    
    % Obtener la lista de píxeles a duplicar en esa fila
    seam = S(k,:);
    
    % Crear un vector idx que mapea las columnas iniciales a las columnas finales
    idx = 1:N*size(img,2);
    idx(seam) = [];
    idx = [idx(1:seam(1)-1) seam idx(seam(1):end)];
    for i = 2:length(seam)
        offset = sum(seam(1:i-1) < seam(i));
        idx = [idx(1:seam(i-1)+offset) seam(i) idx(seam(i-1)+offset+1:end)];
    end
    
    % Rellenar la fila de la nueva imagen ampliada con los píxeles correspondientes
    for i = 1:N
        im2(k,(i-1)*size(img,2)+1:i*size(img,2),:) = img(k,idx(i),:);
    end
end

%%
clear

img = imread('img.jpg');
figure, imshow(img);


n = 144;


S = zeros(size(img, 1), n);
E = energia(img);
for k = 1:n
    M = calcula_M(E);
    seam = find_seam(M);
    S(:, k) = seam;

    E(sub2ind(size(E), seam, repmat(k, size(seam)))) = E(sub2ind(size(E), seam, repmat(k, size(seam)))) * 1.5;
end


hold on;
for k = 1:n
    plot(S(:, k), 'g');
end
hold off;


img2 = zeros(size(img, 1), size(img, 2) + n, size(img, 3), 'uint8');
for k = 1:size(img, 1)

    idx = 1:size(img, 2)+n;
    for j = 1:n

        seam = S(k, j);

        idx = [idx(1:seam) idx(seam) idx(seam+1:end)];
    end
  
    img2(k, :, :) = img(k, idx, :);
end

figure, imshow(img2);

%%

function s = find_seam(M)
    [ALTO, ANCHO] = size(M);
    [~, j] = min(M(1, :));
    s = zeros(ALTO, 1);
    s(1) = j;
    for i = 2:ALTO
        if s(i-1) == 1
            [~, idx] = min(M(i, s(i-1):s(i-1)+1));
            s(i) = s(i-1) + idx - 1;
        elseif s(i-1) == ANCHO
            [~, idx] = min(M(i, s(i-1)-1:s(i-1)));
            s(i) = s(i-1) + idx - 2;
        else
            [~, idx] = min(M(i, s(i-1)-1:s(i-1)+1));
            s(i) = s(i-1) + idx - 2;
        end
    end
end
%%

function M = calcula_M(E)
    
    M = Inf(size(E));
    M(end,:) = E(end,:); 
    
    %
    for i = size(E, 1)-1 : -1 : 1 
        for j = 2 : size(E, 2)-1 
        
            [~, idx] = min(M(i+1, j-1:j+1));
            M(i,j) = E(i,j) + M(i+1, j-2+idx);
        end
    end
   
    M(:,1) = Inf;
    M(:,end) = Inf;
end


%%

function E = energia(im)
   h = fspecial('gaussian', [7 7], 1.5);
    im_smooth = imfilter(im, h, 'symmetric');

   im_diff = abs(im - im_smooth);

   im_detail = sum(im_diff, 3);

   E = 10*log2(1+im_detail);

   E = imresize(E, [size(im, 1), size(im, 2)]);
end

%%
function P = get_proy(xy, uv)
% Construcción de la matriz A
A = zeros(8,9);
for i = 1:4
    A(i*2-1,:) = [xy(1,i) xy(2,i) 1 0 0 0 -xy(1,i)*uv(1,i) -xy(2,i)*uv(1,i) -uv(1,i)];
    A(i*2,:) = [0 0 0 xy(1,i) xy(2,i) 1 -xy(1,i)*uv(2,i) -xy(2,i)*uv(2,i) -uv(2,i)];
end

% Resolución del sistema de ecuaciones lineales
[~, ~, V] = svd(A);
x = V(:,end);

% Construcción de la matriz P
P = reshape(x,[3,3])';

end
%%
function uv = convierte(xy, P)
% Agregar una tercera fila de 1's a la matriz xy para formar una matriz de coordenadas homogéneas 3xN
xyh = [xy; ones(1, size(xy,2))];

% Multiplicar la matriz P por esta matriz de coordenadas homogéneas
uvh = P * xyh;

% Dividir punto a punto las dos primeras filas de la matriz resultante por la tercera fila para obtener las coordenadas de salida uv
uv = uvh(1:2,:) ./ uvh(3,:);

end
%%
function im2 = warp_img(im, iP)
%Calcula la imagen im2 deformada mediante la transformación iP

%Obtenemos las dimensiones de la imagen
[N, M, num_planes_color] = size(im);

%Reservamos espacio para la imagen deformada y las matrices de coordenadas X e Y
im2 = zeros(N, M, num_planes_color);
[X, Y] = meshgrid(1:M, 1:N);

%Calculamos la transformación inversa
iP = inv(iP);

%Barremos las filas de la imagen de salida
for k = 1:N
    %Preparamos la matriz uv para transformar una fila entera
    uv = [1:M; k * ones(1,M)];
    
    %Transformamos las coordenadas de destino (uv) a origen (xy)
    xy = convierte(uv, iP);
    
    %Extraemos las coordenadas x e y
    x = xy(1,:);
    y = xy(2,:);
    
    %Interpolamos la imagen en las coordenadas obtenidas
    for c = 1:num_planes_color
        im2(k,:,c) = interp2(X, Y, im(:,:,c), x, y, 'bilinear');
    end
end

end


%%
function dump_mat(A)
  for k=1:size(A,1)
    fprintf('%9.4f ',A(k,:)); fprintf('\n');  
  end
end
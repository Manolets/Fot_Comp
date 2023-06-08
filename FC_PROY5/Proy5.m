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
%%
% Obtener las coordenadas de las esquinas del cartel en la imagen "billboard.jpg"
dest_corners = [581 1195 579 1193; 269 175 553 651];

src_corners = [1 size(im,2) 1 size(im,2); 1 1 size(im,1) size(im,1)];
P = get_proy(dest_corners, src_corners);

im = im2double(im);
im2 = warp_img(im, inv(P));

imshow(im2);
pause;
im3 = im;

for i = 1:size(im, 1)
    for j = 1:size(im, 2)
        if ~isnan(im2(i, j))
            im3(i, j, :) = im2(i, j, :);
        end
    end
end

imshow(im3);
pause;
im2 = warp_img(im2, inv(P));

imshow(im2);
pause;

for i = 1:size(im, 1)
    for j = 1:size(im, 2)
        if ~isnan(im2(i, j))
            im3(i, j, :) = im2(i, j, :);
        end
    end
end

imshow(im3);

%% 2
xy = [401 643 299 99; 201 296 646 268];
uv = [269 545 276 26; 350 371 744 412];
P = get_nolineal(xy, uv);
dump_mat(P)

%%
% Obtener la matriz de coeficientes P para la transformación no lineal
P = get_nolineal(xy, uv);

% Calcular las coordenadas transformadas usando la función convierte()
uv_transformed = convierte(xy, P);

% Calcular la matriz de diferencias
diff_matrix = uv - uv_transformed;

% Mostrar la matriz de diferencias
disp("Matriz de diferencias:")
disp(diff_matrix);

%%
% Cargar la imagen y convertirla a escala de grises
img = imread('foto.jpg');
img = im2double(img);

% Obtener la matriz de coeficientes de la transformación inversa P_inv
% P_inv = inv(P);
% 
% % Hacer el warping no lineal de la imagen
% img_warp = warp_img(img, P_inv);
% imshow(img_warp)
% Guardar la imagen resultante
% imwrite(img_warp, 'foto_no_lineal.jpg');

%%

% Hacer el warping no lineal de la imagen
img_warp = warp_img(img, P);
imshow(img_warp)
% Guardar la imagen resultante
% imwrite(img_warp, 'foto_no_lineal_directa.jpg');


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
% function uv = convierte(xy, P)
% % Agregar una tercera fila de 1's a la matriz xy para formar una matriz de coordenadas homogéneas 3xN
% xyh = [xy; ones(1, size(xy,2))];
% 
% % Multiplicar la matriz P por esta matriz de coordenadas homogéneas
% uvh = P * xyh;
% 
% % Dividir punto a punto las dos primeras filas de la matriz resultante por la tercera fila para obtener las coordenadas de salida uv
% uv = uvh(1:2,:) ./ uvh(3,:);
% 
% end

function uv = convierte(xy, P)
    if numel(P) == 9 % Transformación lineal
        % Agregar una tercera fila de 1's a la matriz xy para formar una matriz de coordenadas homogéneas 3xN
        xyh = [xy; ones(1, size(xy,2))];

        % Multiplicar la matriz P por esta matriz de coordenadas homogéneas
        uvh = P * xyh;

        % Dividir punto a punto las dos primeras filas de la matriz resultante por la tercera fila para obtener las coordenadas de salida uv
        uv = uvh(1:2,:) ./ uvh(3,:);
    elseif numel(P) == 8 % Transformación no lineal
% Construir matriz de coordenadas homogéneas 4xN
        xyh = [ones(1, size(xy,2)); xy(1,:); xy(2,:); xy(1,:) .* xy(2,:)];

        % Multiplicar la matriz P por esta matriz de coordenadas homogéneas
        uvh = P * xyh;

        % Obtener las coordenadas de salida uv
        uv = uvh(1:2,:);
    else
        error('La matriz P debe tener 8 o 9 elementos.');
    end
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
function P = get_nolineal(xy, uv)

% Construir la matriz H
H = [
    xy(1,1) xy(2,1) 1 xy(1,1)*xy(2,1);
    xy(1,2) xy(2,2) 1 xy(1,2)*xy(2,2);
    xy(1,3) xy(2,3) 1 xy(1,3)*xy(2,3);
    xy(1,4) xy(2,4) 1 xy(1,4)*xy(2,4);
];

% Resolver el sistema lineal para obtener los coeficientes de la transformación
X = H \ uv(1,:)';
Y = H \ uv(2,:)';

% Construir la matriz P
P = [X' ; Y'];

end


%%
function dump_mat(A)
  for k=1:size(A,1)
    fprintf('%9.4f ',A(k,:)); fprintf('\n');  
  end
end
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
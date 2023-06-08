% Número de tomas
P = 9;

% Tiempos de exposición en segundos
T = [1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4];

img = imread('belg_1.jpg');
alto = size(img, 1);
ancho = size(img, 2)

% Inicializar la matriz hdr_data
hdr_data = zeros(alto, ancho, P);

% Bucle para leer las imágenes y extraer datos
for i = 1:P
    % Leer la imagen
    img = imread(sprintf('belg_%d.jpg', i));
    
    % Convertir la imagen a tipo double en el rango [0, 255]
    img_double = im2double(img) * 255;
    
    % Extraer el segundo plano (verde)
    green_channel = img_double(:, :, 2);
    
    % Almacenar el plano verde en la matriz hdr_data
    hdr_data(:, :, i) = green_channel;
end

muestra_HDR(hdr_data, T);

%%
N = 100;

Zdata = extraer_datos(hdr_data, N);

%%
g = solve_G(Zdata, T);

pixels = 0:255;
g_values = g(pixels + 1);

plot(pixels, g_values);
xlabel('Valor de píxel');
ylabel('g(Z)');
title('Función g(Z) en función del valor de píxel');

%%
Zdata100 = extraer_datos(hdr_data, 100);
Zdata10000 = extraer_datos(hdr_data, 10000);
%%
g100 = solve_G(Zdata100, T);
g10000 = solve_G(Zdata10000, T);

plot(pixels, g10000,"Color", [0, 1, 0]);
yyaxis right;
plot(pixels, g100, "Color",[0, 0, 1]);

%%
Zdata10000 = extraer_datos(hdr_data, 10000);
g10000 = solve_G(Zdata10000, T);

log2_R = g10000(hdr_data(:, :, 1) + 1);
R = 2 .^ log2_R;

%%

figure;
imagesc(R);
colormap hot;
colorbar;

%%

R_min = min(R(:));
R_max = max(R(:));
dynamic_range_stops = log2(R_max / R_min);

%%
res = get_log2R(hdr_data, g10000, T);
figure;
imagesc(res);
colormap hot;
colorbar;
%%
d = [ 2 4 121];
g10000(d)

%%

R_min = min(res(:));
R_max = max(res(:));
dynamic_range_stops = log2(R_max / R_min);
%%

    fila = size(hdr_data, 1) / 2;
    
    log2R = res;
    fila = size(hdr_data, 1) / 2;
    log2R_fila = log2R(fila, :);
    log2E = log2R_fila + log2(T)';
    pixeles_verde = hdr_data(fila, :, 2);
    
    figure;
    plot(log2E, pixeles_verde, '.');
    xlabel('log2(E)');
    ylabel('Valor del píxel (canal verde)');
    title('Curva de exposición');
%%
fila = size(hdr_data, 1) / 2;
log2R = res;
figure;
hold on;
log2R_fila = log2R(fila, :);
for i = 1:size(hdr_data, 3)
    log2E = log2R_fila + log2(T(i));
    pixeles_verde = hdr_data(fila, :, i);
    plot(log2E, pixeles_verde, '.');
end

xlabel('log2(E)');
ylabel('Valor del píxel (canal verde)');
title('Curva de exposición');
legend('Toma 1', 'Toma 2', 'Toma 3', 'Toma 4', 'Toma 5', 'Toma 6', 'Toma 7', 'Toma 8', 'Toma 9');

hold off;

%%


P = 9;

T = [1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4];

img = imread('belg_1.jpg');
alto = size(img, 1);
ancho = size(img, 2)

hdr_data_red = zeros(alto, ancho, P);
hdr_data_blue = zeros(alto, ancho, P);
hdr_data_green = zeros(alto, ancho, P);
for i = 1:P
 
    img = imread(sprintf('belg_%d.jpg', i));
    

    img_double = im2double(img) * 255;
    
  
    red_channel = img_double(:, :, 3);
    blue_channel = img_double(:, :, 1);
    green_channel = img_double(:, :, 2);
    
    hdr_data_red(:, :, i) = red_channel;
    hdr_data_blue(:, :, i) = blue_channel;
    hdr_data_green(:, :, i) = green_channel;
end
%%

Zdata10000 = extraer_datos(hdr_data_green, 10000);
g10000 = solve_G(Zdata10000, T);
log2R_green = get_log2R(hdr_data_green, g10000, T);
log2R_red = get_log2R(hdr_data_red, g10000, T);
log2R_blue = get_log2R(hdr_data_blue, g10000, T);

%%
res
%%

log2R_combined = cat(3, log2R_red, log2R_green, log2R_blue);
hdr = 2 .^ log2R_combined;

%%



% Visualización de la imagen HDR
imshow(hdr_rescaled);


%%
hdrwrite(hdr, 'im.hdr');
rgb = tonemap(hdr, 'AdjustSaturation', 3);
imshow(rgb);
%%
hdr = hdrread("im.hdr");
max_radiancia = max(hdr(:));
min_radiancia = min(hdr(:));
rango_dinamico = max_radiancia / min_radiancia;
hdr_rescaled = (hdr - min_radiancia) / (max_radiancia - min_radiancia);
imshow(hdr_rescaled);
%%

im = imread('exp_1.jpg');
im = im2double(im);

N = 5;
p = lap(im, N);

% Verificar los tamaños de los niveles de la pirámide
for k = 1:N
    disp(size(p{k}));
end


%%

N = 5; % Número de niveles de la pirámide laplaciana

% Cargar la imagen "exp_1.jpg"
im = imread('exp_1.jpg');
im = im2double(im);

% Calcular la pirámide laplaciana
p = lap(im, N);

% Verificar los tamaños de los niveles de la pirámide
for k = 1:N
    fprintf('Tamaño del nivel %d: %dx%d\n', k, size(p{k}, 1), size(p{k}, 2));
end

% Visualizar los dos últimos niveles (4º y 5º)
figure;
subplot(1, 2, 1);
imagesc(p{N-1});
title('Nivel 4');
subplot(1, 2, 2);
imagesc(p{N});
title('Nivel 5');
colormap(gray);

%%

im = imread('exp_1.jpg');
im = im2double(im);

N = 5;
p = lap(im, N);

% Verificar los tamaños de los niveles de la pirámide
for k = 1:N
    disp(size(p{k}));
end

% Visualizar los niveles 4º y 5º de la pirámide
figure;
subplot(1, 2, 1);
imagesc(p{4});
axis image;
colormap gray;
title('4º nivel de la pirámide');

subplot(1, 2, 2);
imagesc(p{5});
axis image;
colormap gray;
title('5º nivel de la pirámide');

%%
p1 = lap(im1, 5);
%%
norm([p1{1}(1, 1, 1) p1{1}(1, 1, 2) p1{1}(1, 1, 3)])
%%


im1 = im2double(imread('exp_1.jpg'));
im2 = im2double(imread('exp_2.jpg'));
im3 = im2double(imread('exp_3.jpg'));


N = 5;
p1 = lap(im1, N);
p2 = lap(im2, N);
p3 = lap(im3, N);


p_combined = cell(1, N);

p_combined{N} = (p1{N} + p2{N} + p3{N}) / 3;

for k = 1:N-1
    p_combined{k} = zeros(size(p1{k}));
    
    for i = 1:size(p1{k}, 1)
        for j = 1:size(p1{k}, 2)
            detail_norms = [norma(p1, k, i, j), norma(p2, k, i, j),norma(p3, k, i, j)];
            [~, max_idx] = max(detail_norms);
            p_combined{k}(i, j, :) = p1{k}(i, j, :);
            
            if max_idx == 2
                p_combined{k}(i, j, :) = p2{k}(i, j, :);
            elseif max_idx == 3
                p_combined{k}(i, j, :) = p3{k}(i, j, :);
            end
        end
    end
end

% Visualizar los niveles 4º y 5º de la pirámide combinada
figure;
subplot(1, 2, 1);
imagesc(p_combined{4});
axis image;
colormap gray;
title('4º nivel de la pirámide combinada');

subplot(1, 2, 2);
imagesc(p_combined{5});
axis image;
colormap gray;
title('5º nivel de la pirámide combinada');

%%
HDR = p_combined{N}; 
for k = N-1:-1:1
    HDR = imresize(HDR, 2);  
    HDR = HDR + p_combined{k};
end


%%
% Calculating the absolute difference between the original image and the reconstructed HDR image
diff_image = abs(im1 - HDR);

% Display the difference image
imshow(diff_image);

%%

max(diff_image(:))

%%

v0 = min(HDR(:));
v1 = max(HDR(:));
HDR_rescaled = (HDR - v0) / (v1 - v0);
HSV = rgb2hsv(HDR_rescaled);
HSV(:, :, 3) = adapthisteq(HSV(:, :, 3), 'ClipLimit', 0.01);
HSV(:, :, 2) = HSV(:, :, 2).^0.75;  % Enhance saturation
HDR_final = hsv2rgb(HSV);
%%

imshow(HDR_final);

%%

next = fusion( "belg_3.jpg", "belg_5.jpg", "belg_7.jpg");
imshow(next);
%%

function res = norma(cell, k, i, j)
    res = norm([cell{k}(i, j, 1) cell{k}(i, j, 2) cell{k}(i, j, 3)]);
end

function log2R = get_log2R(hdr_data, g, T)
    [height, width, ~] = size(hdr_data);
    log2R = zeros(height, width);
    
    M = 256;
    t=(1:M)'/(M+1);
    w=(t.*(1-t)).^2;
    w=w/max(w);


    for y = 1:height
        for x = 1:width
            Z = squeeze(hdr_data(y, x, :));
            Z = round(Z, 0)+1;
            g(Z)
            g_Z = g(Z) - log2(T)';
            weights = w(Z);
            weights = weights / sum(weights); 
            log2R(y, x) = sum( weights.^g_Z);
        end
    end
end


%%


function muestra_HDR(hdr_set,T) 

if nargin==0, load HDR_data, end

P = length(T);


figure(1); %colormap(gray(256));
set(gcf,'Pos',[510 50 1400 700]);
DY=0.99-0.32; DX=0.005; ANCHO=0.18; ALTO=0.32;

fot=zeros(1,9); cont=1;
for k=3:-1:1, 
  DX=0.01;   
  for j=3:-1:1 
  fot(cont)=axes('Position',[DX DY ANCHO ALTO]); cont=cont+1;
  imshow(hdr_set(:,:,3*(k-1)+j)/255); colormap(gray);
  set(gca,'Xtick',[],'Ytick',[],'Xcolor','r','Ycolor','r');
  set(gca,'LineWidth',2);
  DX=DX+ANCHO+0.007;  
  %pause(0.5)
  end
  DY=DY-ALTO-0.01;
end  

DY=0.2; 
%DX=DX+0.02, %DY=0.2;
DX=0.59;
ANCHO=0.4; ALTO=0.50;
ejes=axes('Position',[DX DY ANCHO ALTO]);
pix=log2(T)*NaN;
pl=plot(log2(T),pix,'ko:','LineWidth',2); 
set(gca,'Ylim',[-5 260]);
title('Valor del píxel (gris)');
xlabel('log_2(T) de cada toma');

cmap=jet(8);

r=0;
b=1; idx=1; [j i b]=ginput(1);
while(b~=3)    
  cc=cmap(r+1,:);    
   for k=1:9,
    axes(fot(k));   
    hold on;   
    pp=plot(j,i,'o');
    set(pp,'MarkerEdgeColor',cc,'MarkerFaceColor',cc); 
    hold off
   end
  j=round(j); i=round(i); 
  values = reshape(hdr_set(i,j,:),1,P); 
  axes(ejes);
  hold on; plot(log2(T),(values),'o:','Color',cc,'LineWidth',2);
  %set(pl,'Ydata',mean(values));  
  r=mod(r+1,8);
  [j i b]=ginput(1);
end  


end


%%
function Zdata = extraer_datos(hdr_data, N)
    % Obtener dimensiones de la imagen
    [alto, ancho, P] = size(hdr_data);
    
    % Inicializar matriz Zdata
    Zdata = zeros(P, N);
    
    % Mostrar la imagen de la 5ª toma
    figure;
    imshow(hdr_data(:, :, 5), []);
    hold on;
    
    % Bucle para escoger y extraer datos en N puntos
    for n = 1:N
        % Escoger coordenadas (i, j) aleatorias dentro de la imagen
        i = floor(rand * alto) + 1;
        j = floor(rand * ancho) + 1;
        
        % Extraer los datos de las P tomas en la posición (i, j)
        pixel_values = squeeze(hdr_data(i, j, :));
        
        % Verificar que los valores sean crecientes
        if all(diff(pixel_values) >= 0)
            % Guardar los valores en la columna correspondiente de Zdata
            Zdata(:, n) = pixel_values;
            
            % Dibujar un punto verde en la posición seleccionada
            plot(j, i, 'g.');
        end
    end
    
    hold off;
end

%%
function g = solve_G(Zdata, T)
    N = size(Zdata, 2);
    P = size(Zdata, 1);
    Neq = N * (P - 1) + 255;
    b = zeros(Neq, 1);
    num_nonzeros = 2 * N * (P - 1) + 3 * 254 + 1;
    i = zeros(num_nonzeros, 1);
    j = zeros(num_nonzeros, 1);
    v = zeros(num_nonzeros, 1);
    
    En = log2(T);

    M = 256; 
    t=(1:M)'/(M+1); 
    w=(t.*(1-t)).^2; 
    w=w/max(w);
    
    eq_count = 0; % Contador de ecuaciones
    entry = 1;
    for n = 1:N
        Zn = Zdata(:, n);
        [~, index] = min(abs(Zn - 128)); % Valor más cercano a 128
        Zref = Zn(index);
        if index == 1
            Zn = Zn(2:P);
        end 
        if index == P
            Zn = Zn(1:P-1);
        end
        if index ~= 1 && index ~= P
            Zn = Zn([1:index - 1, index + 1:P]);
        end

        for p = 1:P-1
            eq_count = eq_count + 1;
            
            W = 1;% sqrt(w(Zn(p) + 1)*w(Zref + 1));
            
            i(entry) = eq_count;
            j(entry) = Zn(p) + 1;
            v(entry) = 1 * W;
            
            entry = entry + 1;

            i(entry) = eq_count;
            j(entry) = Zref + 1;
            v(entry) = -1 * W;
            b(p + (n-1)*(P-1)) = (En(p) / En(index)) * W;
            
            entry = entry + 1;

        end
    end

    lambda = 0;
    for z = 1:254
        eq_count = eq_count + 1;
        i(entry) = eq_count;
        j(entry) = z;
        v(entry) = -1 * lambda;
        
        entry = entry + 1;

        i(entry) = eq_count;
        j(entry) = z + 1;
        v(entry) = 2 * lambda;
    
        entry = entry + 1;

        i(entry) = eq_count;
        j(entry) = z + 2;
        v(entry) = -1 * lambda;

        entry = entry + 1;
    end
    
    eq_count = eq_count + 1;
    i(entry) = eq_count;
    j(entry) = 129; % Índice correspondiente a g128
    v(entry) = 1  * lambda;

    H = sparse(i, j, v, Neq, 256);

    disp(size(H)); % Dimensiones de H
    disp(sum(H ~= 0)); % Número de elementos no nulos en cada columna
    plot(sum(H ~= 0)); % Gráfico de la distribución de elementos no nulos

    disp(sum(H(:, 50) ~= 0)); % Número de veces que aparece g50 en las ecuaciones
    disp(sum(H(:, 200) ~= 0)); % Número de veces que aparece g200 en las ecuaciones

    g = H\ b;
end


function p = lap(im, N)
    p = cell(1, N);
    
    for k = 1:N-1
        p{k} = im;
        im = imresize(im, 0.5);
        im2 = imresize(im, 2);
        p{k} = p{k} - im2;
    end
    
    p{N} = im;
end

